# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "zarr>=3", "obstore", "duckdb>=2.0.1", "psutil", "shapely", "scipy"]
# ///
"""Does the fuel around where a fire STARTS say how big it gets? The lower 48, 2020 to 2026.

burn_fuel.py reads the fuel inside each final perimeter, which is known only after the fire. This reads what a
fire could see at its start: the national volume head (models/fuel_volume_conus.json) on the AlphaEarth embeddings
of the year BEFORE the fire, within 1, 5 and 10 km of the point of origin.

Fires: WFIGS wildfire records (aef-landcover-fosberg-cog/data/ignition/wfigs.parquet) with a size, in six size
classes; every fire of 10,000 acres and up, and up to --per-class drawn at random from each smaller class, so the
small fires that stayed small are in the table too.

Reading: the mosaic is stored in chunks of 256 pixels (about 2.3 km). Every chunk touching the 1 km circle is
read, one chunk in four out to 5 km and one in sixteen out to 10 km, so the 5 and 10 km figures are an even
sample of that ground, not all of it. Pixels are averaged to 160 m blocks: mean volume and the shares that read
none / tree / shrub / herb. A figure "within 5 km" is the mean of the 0 to 1 km and 1 to 5 km rings weighed by
their areas (and the same out to 10 km), so the thinner sampling far out does not tilt it. Also kept per fire: mean volume in 8 compass sectors 2 to 10 km out (sector 0 is
north, clockwise), for the direction test.

Reported: fuel around the start by size class; Spearman of fuel with log acres; and of 1,000 starts on each band
of fuel, how many pass 1,000, 10,000 and 100,000 acres (classes weighted back to their real numbers).

RAM: a chunk is 4 MB of int8 and 17 MB as float32; about 1.5 GB at 32 readers (0.8 GB measured at 16). Stops past --max-gb.
Writes results/start_fuel.json (one row per fire, about 1 MB).

    uv run scripts/start_fuel.py [--per-class 400] [--threads 16] [--limit N]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.stats import spearmanr

os.environ.setdefault("FUEL_HEAD", "fuel_volume_conus.json")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_fuel as bf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WFIGS = ROOT.parent / "aef-landcover-fosberg-cog" / "data" / "ignition" / "wfigs.parquet"
CH, B = 256, 16
EDGES = [0, 10, 100, 1000, 10000, 100000, 1e9]
CLASSES = ["under 10 acres", "10 to 100", "100 to 1k", "1k to 10k", "10k to 100k", "100k and up"]
BANDS = [(0, 0.5, "under 0.5 m"), (0.5, 2, "0.5 to 2 m"), (2, 5, "2 to 5 m"), (5, 1e9, "over 5 m")]
RADII = (1, 5, 10)


def fires(per_class, seed=0):
    """The sample: one row per fire, with the weight that takes its class back to its real number."""
    import duckdb
    d = duckdb.connect().execute(f"""
        SELECT IrwinID AS irwin, IncidentName AS name, POOState AS state, FireCause AS cause,
               FireDiscoveryDateTime AS discovered, year(FireDiscoveryDateTime) AS year,
               coalesce(FinalAcres, IncidentSize) AS acres, geom_lon AS lon, geom_lat AS lat
        FROM '{WFIGS}' WHERE POOState NOT IN ('US-AK', 'US-HI') AND geom_lat BETWEEN 24.5 AND 49.4
          AND geom_lon BETWEEN -124.8 AND -66.9 AND coalesce(FinalAcres, IncidentSize) > 0
          AND year(FireDiscoveryDateTime) BETWEEN 2020 AND 2026 ORDER BY IrwinID""").df()
    d = d.drop_duplicates("irwin").reset_index(drop=True)
    d["cls"] = pd.cut(d.acres, EDGES, labels=CLASSES, right=False).astype(str)
    rng = np.random.default_rng(seed)
    keep = []
    for cls, g in d.groupby("cls"):
        n = len(g) if cls in CLASSES[4:] else min(len(g), per_class)
        pick = g.iloc[np.sort(rng.choice(len(g), n, replace=False))].copy()
        pick["weight"] = len(g) / n
        keep.append(pick)
    return pd.concat(keep, ignore_index=True)


def chunks_for(lon, lat):
    """The chunks (row, col) read for one start: all within 1 km, one in four to 5 km, one in sixteen to 10 km."""
    kx = 111.2 * np.cos(np.radians(lat))
    out = set()
    for km, step in ((1, 1), (5, 2), (10, 4)):
        c0, c1 = int((lon - km / kx - bf.X0) / bf.RES_DEG) // CH, int((lon + km / kx - bf.X0) / bf.RES_DEG) // CH
        r0, r1 = int((bf.Y0 - lat - km / 111.2) / bf.RES_DEG) // CH, int((bf.Y0 - lat + km / 111.2) / bf.RES_DEG) // CH
        out |= {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if r % step == 0 and c % step == 0}
    return out


def read(job):
    """One chunk of one year as 160 m blocks: lon, lat, volume, none, tree, shrub, herb, pixels."""
    ti, r, c = job
    arr, _ = bf.mosaic()
    for attempt in range(5):
        try:
            e = np.asarray(arr[ti, :, r * CH:(r + 1) * CH, c * CH:(c + 1) * CH])
            break
        except Exception as err:  # noqa: BLE001
            if attempt == 4:
                print(f"FAILED {ti}_{r}_{c}: {err}", flush=True)
                return job, None
            time.sleep(2 + 4 * attempt)
    q = e.reshape(64, -1).T
    ok = q[:, 0] != -128
    if not ok.any():
        return job, np.empty((0, 8), np.float32)
    vol, life = np.zeros(len(q), np.float32), np.full(len(q), -1, np.int8)
    vol[ok], life[ok] = bf.volume(q[ok])
    n = CH // B
    blk = lambda a: a.reshape(n, B, n, B).sum(axis=(1, 3))  # noqa: E731
    cnt = blk(ok.reshape(CH, CH).astype(np.float32))
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = [blk(vol.reshape(CH, CH)) / cnt] + [blk((life == k).reshape(CH, CH).astype(np.float32)) / cnt for k in range(4)]
    yy = bf.Y0 - (r * CH + (np.arange(n) + 0.5) * B) * bf.RES_DEG
    xx = bf.X0 + (c * CH + (np.arange(n) + 0.5) * B) * bf.RES_DEG
    X, Y = np.meshgrid(xx, yy)
    out = np.stack([X, Y] + cols + [cnt], -1).reshape(-1, 8)
    return job, out[out[:, 7] > 0].astype(np.float32)


def per_fire(f, blocks):
    """The fuel around one start from the blocks of its chunks."""
    kx = 111.2 * np.cos(np.radians(f.lat))
    dx, dy = (blocks[:, 0].astype(np.float64) - f.lon) * kx, (blocks[:, 1].astype(np.float64) - f.lat) * 111.2
    dist, w = np.hypot(dx, dy), blocks[:, 7]
    row = {}
    rings = [(dist < 1), (dist >= 1) & (dist < 5), (dist >= 5) & (dist < 10)]
    area = [1.0, 24.0, 75.0]
    if not rings[0].any():
        return None
    cols = dict(vol=blocks[:, 2], none=blocks[:, 3], tree=blocks[:, 4], shrub=blocks[:, 5], herb=blocks[:, 6],
                heavy=(blocks[:, 2] > 2.0).astype(np.float32))
    for col, v in cols.items():
        num = den = 0.0
        for km, m, a_ in zip(RADII, rings, area):
            if m.any():  # a ring with nothing read (all sea) adds nothing
                num, den = num + a_ * float((v[m] * w[m]).sum() / w[m].sum()), den + a_
            row[f"{col}_{km}"] = round(num / den, 3)
    row["blocks_10"] = int((dist < 10).sum())
    sector = (np.floor((np.degrees(np.arctan2(dx, dy)) + 22.5) % 360 / 45)).astype(int)
    ring = (dist >= 2) & (dist < 10)
    row["sectors"] = [round(float((blocks[ring & (sector == s), 2] * w[ring & (sector == s)]).sum()
                                  / max(w[ring & (sector == s)].sum(), 1e-9)), 3) if (ring & (sector == s)).any() else None
                      for s in range(8)]
    return row


def boot(fn, groups, n, rng):
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    out = [fn(np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])) for _ in range(n)]
    return [round(float(fn(np.arange(len(groups)))), 3), round(float(np.nanpercentile(out, 2.5)), 3),
            round(float(np.nanpercentile(out, 97.5)), 3)]


def report(df):
    rng = np.random.default_rng(0)
    out = dict(fires=int(len(df)), years=[int(df.year.min()), int(df.year.max())], by_class={}, spearman={}, per_1000={})
    for cls in CLASSES:
        g = df[df.cls == cls]
        if len(g) == 0:
            continue
        out["by_class"][cls] = dict(fires=int(len(g)), stands_for=int(round(g.weight.sum())),
                                    **{f"vol_{km}km_median": round(float(g[f"vol_{km}"].median()), 2) for km in RADII},
                                    **{f"heavy_{km}km_mean": round(float(g[f"heavy_{km}"].mean()), 2) for km in (5, 10)},
                                    **{f"{k}_5km": round(float(g[f"{k}_5"].mean()), 2) for k in ("tree", "shrub", "herb", "none")})
    la = np.log10(df.acres.to_numpy())
    place = (np.floor(df.lon) * 1000 + np.floor(df.lat)).to_numpy()
    sets = {"all sampled fires (equal numbers per class up to 10k acres)": np.ones(len(df), bool),
            "100 acres and up": (df.acres >= 100).to_numpy(), "1,000 acres and up": (df.acres >= 1000).to_numpy()}
    for name, sel in sets.items():
        res = {}
        for col in [f"vol_{km}" for km in RADII] + ["heavy_10", "tree_5", "shrub_5", "herb_5", "none_5"]:
            s = df[col].to_numpy()
            res[col] = boot(lambda i: spearmanr(s[sel][i], la[sel][i]).statistic, place[sel], 300, rng)
        out["spearman"][name] = dict(fires=int(sel.sum()), **res)
    out["spearman_by_year"] = {int(y): round(float(spearmanr(g.vol_5, np.log10(g.acres)).statistic), 3) for y, g in df.groupby("year")}
    for km in (1, 5, 10):
        res = {}
        for lo, hi, name in BANDS:
            g = df[(df[f"vol_{km}"] >= lo) & (df[f"vol_{km}"] < hi)]
            wsum = g.weight.sum()
            res[name] = dict(sampled=int(len(g)), stands_for=int(round(wsum)),
                             **{f"past_{t:,}": round(float(1000 * g.weight[g.acres >= t].sum() / max(wsum, 1e-9)), 2)
                                for t in (100, 1000, 10000, 100000)},
                             big_fires=int((g.acres >= 10000).sum()))
        out["per_1000"][f"fuel within {km} km"] = res
    pd.set_option("display.width", 250)
    print(f"\n{len(df)} fires, {out['years'][0]} to {out['years'][1]}\n\nfuel around the start, by final size")
    print(pd.DataFrame(out["by_class"]).T.to_string())
    print("\nSpearman with log acres (95 % interval, resampling 1 degree places)")
    for name, res in out["spearman"].items():
        print(f"  {name}: {res['fires']} fires")
        print("    " + "  ".join(f"{k} {v[0]:+.2f} ({v[1]:+.2f} to {v[2]:+.2f})" for k, v in res.items() if k != "fires"))
    print("  by year, volume within 5 km: " + "  ".join(f"{y} {v:+.2f}" for y, v in out["spearman_by_year"].items()))
    for name, res in out["per_1000"].items():
        print(f"\nof 1,000 starts, how many pass each size, by {name}")
        print(pd.DataFrame(res).T.to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-gb", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    fr = fires(a.per_class)
    if a.limit:
        fr = fr.sample(a.limit, random_state=0).reset_index(drop=True)
    _, times = bf.mosaic()
    fr = fr[(fr.year - 1).isin(times)].reset_index(drop=True)
    want = [chunks_for(f.lon, f.lat) for f in fr.itertuples()]
    jobs = sorted({(times.index(f.year - 1), r, c) for f, w in zip(fr.itertuples(), want) for r, c in w})
    print(f"{len(fr)} fires ({fr.cls.value_counts().reindex(CLASSES).to_dict()}), {len(jobs)} chunks to read", flush=True)
    proc, got, t0, failed = psutil.Process(os.getpid()), {}, time.time(), 0
    with ThreadPoolExecutor(a.threads) as ex:
        for i, (job, blocks) in enumerate(ex.map(read, jobs)):
            if blocks is None:
                failed += 1
            else:
                got[job] = blocks
            gb = proc.memory_info().rss / 1e9
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}", flush=True)
                os._exit(2)
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"{i + 1}/{len(jobs)} chunks  {el / 60:.1f} min  {gb:.1f} GB  "
                      f"about {el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min left", flush=True)
    rows = []
    for f, w in zip(fr.itertuples(), want):
        ti = times.index(f.year - 1)
        parts = [got[(ti, r, c)] for r, c in w if (ti, r, c) in got and len(got[(ti, r, c)])]
        row = per_fire(f, np.concatenate(parts)) if parts else None
        if row is None:
            continue
        rows.append(dict(irwin=f.irwin, name=f.name, state=f.state, cause=f.cause, year=int(f.year),
                         discovered=f.discovered.isoformat(), acres=float(f.acres), lon=round(f.lon, 5), lat=round(f.lat, 5),
                         cls=f.cls, weight=round(float(f.weight), 3), **row))
    df = pd.DataFrame(rows)
    out = report(df)
    out["failed_chunks"] = failed
    out["rows"] = rows
    (ROOT / "results" / "start_fuel.json").write_text(json.dumps(out))
    print(f"\n{failed} chunks failed; {len(df)} of {len(fr)} fires kept; {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
