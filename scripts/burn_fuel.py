# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "zarr>=3", "obstore", "duckdb>=2.0.1", "psutil", "shapely"]
# ///
"""Where fires burned, 2020 to 2026: the fuel the year BEFORE each fire, inside its perimeter and just outside.

Perimeters: aef-hls-s1-fire-knn/data/fire_test/perimeters.parquet (NIFC, California, 100 acres and up). For a fire
of year Y the volume head (models/fuel_volume.json, as in the California bake) runs on the AlphaEarth embeddings of
Y - 1 over the perimeter and 5 km around it, so the fire's own scar is never in the fuel. Pixels are averaged to
100 m blocks: mean volume, and the shares that read none / tree / shrub / herb.

Per fire: the mean inside, in the ring 0 to 2 km out and the ring 2 to 5 km out, and the chance a burned block
carries more volume than a block 2 to 5 km out. Then summed up by fire size and by year, and next to California
as a whole (the 2025 bake).

Nothing is kept but results/burn_fuel.json (under 1 MB). RAM: one window is 67 MB of int8 and its float copy,
about 0.5 GB a reader; under 5 GB at 8 readers. Stops past --max-gb.

    uv run scripts/burn_fuel.py [--years 2020-2026] [--threads 8]
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
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_fuel as bf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PERIMS = ROOT.parent / "aef-hls-s1-fire-knn" / "data" / "fire_test" / "perimeters.parquet"
WIN, B = 1024, 10


def read(job):
    """One window of one year as 100 m blocks: lon, lat, volume, none, tree, shrub, herb."""
    ti, r, c = job
    arr, _ = bf.mosaic()
    for attempt in range(5):
        try:
            e = np.asarray(arr[ti, :, r * WIN:(r + 1) * WIN, c * WIN:(c + 1) * WIN])
            break
        except Exception as err:  # noqa: BLE001
            if attempt == 4:
                print(f"FAILED {r}_{c}: {err}", flush=True)
                return np.empty((0, 7), np.float32)
            time.sleep(3 + 5 * attempt)
    q = e.reshape(64, -1).T
    ok = q[:, 0] != -128
    if not ok.any():
        return np.empty((0, 7), np.float32)
    vol, life = np.full(len(q), np.nan, np.float32), np.full(len(q), -1, np.int8)
    vol[ok], life[ok] = bf.volume(q[ok])
    n = WIN // B * B
    v = vol.reshape(WIN, WIN)[:n, :n].reshape(n // B, B, n // B, B)
    lf = life.reshape(WIN, WIN)[:n, :n].reshape(n // B, B, n // B, B)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        cols = [np.nanmean(v, axis=(1, 3))] + [(lf == k).mean(axis=(1, 3)) / np.maximum((lf >= 0).mean(axis=(1, 3)), 1e-9)
                                              for k in (0, 1, 2, 3)]
    yy = bf.Y0 - (r * WIN + (np.arange(n // B) + 0.5) * B) * bf.RES_DEG
    xx = bf.X0 + (c * WIN + (np.arange(n // B) + 0.5) * B) * bf.RES_DEG
    X, Y = np.meshgrid(xx, yy)
    out = np.stack([X, Y] + cols, -1).reshape(-1, 7)
    return out[np.isfinite(out[:, 2])].astype(np.float64)


def auc(a, b, cap=20000, rng=np.random.default_rng(0)):
    if len(a) > cap:
        a = rng.choice(a, cap, replace=False)
    if len(b) > cap:
        b = rng.choice(b, cap, replace=False)
    r = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020-2026")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=8.0)
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-"))
    per = pd.read_parquet(PERIMS)
    per = per[(per.year >= y0) & (per.year <= y1)].reset_index(drop=True)
    per["g"] = shapely.from_wkb(per.geom.to_numpy())
    _, times = bf.mosaic()
    proc, rows, t0 = psutil.Process(os.getpid()), [], time.time()
    for year, fires in per.groupby("year"):
        if year - 1 not in times:
            print(f"{year}: no embeddings for {year - 1}", flush=True)
            continue
        zones, want = [], set()
        for f in fires.itertuples():
            lat = f.g.centroid.y
            k = np.array([111.2 * np.cos(np.radians(lat)), 111.2])
            km = shapely.make_valid(shapely.transform(f.g, lambda c, k=k: c * k))
            r2, r5 = km.buffer(2), km.buffer(5)
            zones.append((f, k, km, r2, r5))
            lo0, la0, lo1, la1 = (np.array(r5.bounds).reshape(2, 2) / k).ravel()
            c0, c1 = int((lo0 - bf.X0) / bf.RES_DEG) // WIN, int((lo1 - bf.X0) / bf.RES_DEG) // WIN
            rr0, rr1 = int((bf.Y0 - la1) / bf.RES_DEG) // WIN, int((bf.Y0 - la0) / bf.RES_DEG) // WIN
            for r in range(rr0, rr1 + 1):
                for c in range(c0, c1 + 1):
                    box = shapely.box((bf.X0 + c * WIN * bf.RES_DEG) * k[0], (bf.Y0 - (r + 1) * WIN * bf.RES_DEG) * k[1],
                                      (bf.X0 + (c + 1) * WIN * bf.RES_DEG) * k[0], (bf.Y0 - r * WIN * bf.RES_DEG) * k[1])
                    if r5.intersects(box):
                        want.add((r, c))
        jobs = [(times.index(year - 1), r, c) for r, c in sorted(want)]
        print(f"{year}: {len(fires)} fires, {len(jobs)} windows of {year - 1} embeddings", flush=True)
        parts = []
        with ThreadPoolExecutor(a.threads) as ex:
            for i, g in enumerate(ex.map(read, jobs)):
                parts.append(g)
                gb = proc.memory_info().rss / 1e9
                if gb > a.max_gb:
                    print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}", flush=True)
                    os._exit(2)
                if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                    print(f"  {i + 1}/{len(jobs)} windows  {time.time() - t0:.0f} s  {gb:.1f} GB", flush=True)
        g = np.concatenate(parts)
        del parts
        for f, k, km, r2, r5 in zones:
            lo0, la0, lo1, la1 = (np.array(r5.bounds).reshape(2, 2) / k).ravel()
            d = g[(g[:, 0] >= lo0) & (g[:, 0] <= lo1) & (g[:, 1] >= la0) & (g[:, 1] <= la1)]
            x, y = d[:, 0] * k[0], d[:, 1] * k[1]
            inside = shapely.contains_xy(km, x, y)
            in2 = shapely.contains_xy(r2, x, y) & ~inside
            in5 = shapely.contains_xy(r5, x, y) & ~inside & ~in2
            if inside.sum() < 20 or in5.sum() < 20:
                continue
            row = dict(year=int(year), name=f.name, acres=float(f.acres), blocks=int(inside.sum()),
                       auc_vol=round(auc(d[inside, 2], d[in5, 2]), 3))
            for z, m in (("in", inside), ("r2", in2), ("r5", in5)):
                for j, col in ((2, "vol"), (3, "none"), (4, "tree"), (5, "shrub"), (6, "herb")):
                    row[f"{col}_{z}"] = round(float(d[m, j].mean()), 3) if m.any() else None
            rows.append(row)
        del g
    df = pd.DataFrame(rows)

    def summary(d):
        w = d.blocks.to_numpy()
        return dict(fires=int(len(d)), acres=round(float(d.acres.sum())),
                    vol_in_median=round(float(d.vol_in.median()), 2), vol_r5_median=round(float(d.vol_r5.median()), 2),
                    vol_in_by_area=round(float((d.vol_in * w).sum() / w.sum()), 2), tree_in_by_area=round(float((d.tree_in * w).sum() / w.sum()), 2),
                    shrub_in_by_area=round(float((d.shrub_in * w).sum() / w.sum()), 2), herb_in_by_area=round(float((d.herb_in * w).sum() / w.sum()), 2),
                    none_in_by_area=round(float((d.none_in * w).sum() / w.sum()), 3), none_r5_mean=round(float(d.none_r5.mean()), 3),
                    vol_higher_inside_than_r2=round(float((d.vol_in > d.vol_r2).mean()), 2),
                    vol_higher_inside_than_r5=round(float((d.vol_in > d.vol_r5).mean()), 2),
                    less_bare_inside_than_r5=round(float((d.none_in < d.none_r5).mean()), 2),
                    auc_vol_median=round(float(d.auc_vol.median()), 3),
                    share_over_2m=round(float((d.vol_in > 2.13).mean()), 2), acres_share_over_2m=round(float(d.acres[d.vol_in > 2.13].sum() / d.acres.sum()), 2))

    size = pd.cut(df.acres, [0, 1000, 10000, 100000, 1e9], labels=["100 to 1k acres", "1k to 10k", "10k to 100k", "100k and up"])
    out = dict(all=summary(df), by_size={str(k): summary(d) for k, d in df.groupby(size, observed=True)},
               by_year={str(k): summary(d) for k, d in df.groupby("year")},
               largest=df.sort_values("acres", ascending=False).head(25).to_dict("records"), fires=df.to_dict("records"))
    (ROOT / "results" / "burn_fuel.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 250)
    for name, part in (("all", {"all": out["all"]}), ("by size", out["by_size"]), ("by year", out["by_year"])):
        print(f"\n{name}\n{pd.DataFrame(part).T.to_string()}")
    print("\nlargest\n" + df.sort_values("acres", ascending=False).head(25)[
        ["year", "name", "acres", "vol_in", "vol_r2", "vol_r5", "tree_in", "shrub_in", "herb_in", "none_in", "none_r5", "auc_vol"]].to_string(index=False))


if __name__ == "__main__":
    main()
