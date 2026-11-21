# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "duckdb>=2.0.1", "scikit-learn", "scipy"]
# ///
"""What the baked fuel says about a GOES fire event once it is seen: is it a recorded wildfire, and how big does it get?

goes_overlap.py asks where and when fires start. This asks what follows. Events and labels: aef-hls-s1-fire-knn
(data/goes/events_<year>.parquet, event_px_<year>.parquet for the first box, event_fuel_<year>.parquet for that
project's own fuel readings, scored here on the same rows for comparison).

Only 2026: the fuel is from 2025 embeddings, so an earlier fire's scar would already be in it.

Fuel read from data/parts (A5 resolution 15, summed to resolution 11, about a GOES pixel):
  in      the cells under the event's first-box pixels
  around  the cells 5 km out in 8 directions, less the ones under the box
  each as mean volume (m), and the share of 10 m pixels that are tree, shrub, herb, or any of the three

Scores, intervals 95 % resampling 0.1 degree places:
  wildfire   AUC, recorded wildfire (perimeter or start) against unmatched; one reading at a time, and logistic fits
             scored out of fold by place: GOES first scan (power, class, pixels) alone and with fuel added
  size       recorded wildfires, one row per fire (its first event): Spearman with log acres, AUC past 100 acres

Under 2 GB of RAM. Writes results/goes_fuel_role_<year>.json (a few KB).

    uv run scripts/goes_fuel_role.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
GOES = ROOT.parent / "aef-hls-s1-fire-knn" / "data" / "goes"
RES = 11
MINE = ["vol_in", "vol_around", "veg_in", "veg_around", "tree_in", "shrub_in", "herb_in"]
THEIRS = ["knn_in", "knn_around", "head_in", "head_around", "lf_in", "lf_around"]
GOES_X = ["g_power", "g_class", "g_px"]


def boot(fn, groups, n, rng):
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    out = []
    for _ in range(n):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        try:
            out.append(fn(i))
        except ValueError:
            pass
    return [round(float(fn(np.arange(len(groups)))), 3), round(float(np.nanpercentile(out, 2.5)), 3),
            round(float(np.nanpercentile(out, 97.5)), 3)]


def oof(X, y, groups, model):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, groups=groups):
        m = make_pipeline(StandardScaler(), model()).fit(X[tr], y[tr])
        p[te] = m.decision_function(X[te]) if hasattr(m[-1], "decision_function") else m.predict(X[te])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET allow_extension_repositories = 'allowed'")
    con.execute("LOAD a5 FROM query_farm")
    con.execute(f"""CREATE TABLE fuel AS SELECT a5_cell_to_parent(cell, {RES}) AS cell, sum(vol)::DOUBLE AS vol,
                    sum(n)::DOUBLE AS n, sum(tree)::DOUBLE AS tree, sum(shrub)::DOUBLE AS shrub, sum(herb)::DOUBLE AS herb
                    FROM '{ROOT}/data/parts/*.parquet' GROUP BY 1""")
    ev = pd.read_parquet(GOES / f"event_fuel_{a.year}.parquet")
    px = pd.read_parquet(GOES / f"event_px_{a.year}.parquet")[["event", "lon", "lat"]]
    ring = []
    for b in np.arange(8) * np.pi / 4:
        ring.append(pd.DataFrame(dict(event=ev.event, lon=ev.lon + 5 * np.sin(b) / (111.2 * np.cos(np.radians(ev.lat))),
                                      lat=ev.lat + 5 * np.cos(b) / 111.2)))
    ring = pd.concat(ring, ignore_index=True)  # noqa: F841 (read by duckdb)
    agg = """sum(f.vol) / sum(f.n) AS vol, sum(f.tree + f.shrub + f.herb) / sum(f.n) AS veg, sum(f.tree) / sum(f.n) AS tree,
             sum(f.shrub) / sum(f.n) AS shrub, sum(f.herb) / sum(f.n) AS herb"""
    con.execute(f"CREATE TABLE cin AS SELECT DISTINCT event, a5_lonlat_to_cell(lon, lat, {RES}) AS cell FROM px")
    con.execute(f"""CREATE TABLE cout AS SELECT DISTINCT event, a5_lonlat_to_cell(lon, lat, {RES}) AS cell FROM ring
                    EXCEPT SELECT event, cell FROM cin""")
    fin = con.execute(f"SELECT c.event, {agg} FROM cin c JOIN fuel f USING (cell) GROUP BY 1").df().add_suffix("_in")
    fout = con.execute(f"SELECT c.event, {agg} FROM cout c JOIN fuel f USING (cell) GROUP BY 1").df().add_suffix("_around")
    d = ev.merge(fin, left_on="event", right_on="event_in").merge(fout, left_on="event", right_on="event_around")
    for k in ("vol_in", "vol_around"):
        d[k] = np.log(0.02 + d[k])
    d["g_power"] = np.log1p(d.first_power.fillna(0).clip(lower=0))
    d["g_class"] = d.first_strength.astype(float)
    d["g_px"] = np.log(d.first_px.clip(lower=1))
    d["place"] = np.floor(d.lon * 10).astype(int) * 10000 + np.floor(d.lat * 10).astype(int)
    d["wild"] = (d.label != "unmatched").astype(int)
    d = d.dropna(subset=MINE + THEIRS).reset_index(drop=True)
    print(f"{len(ev)} events in {a.year}, {len(d)} with every fuel reading, {int(d.wild.sum())} recorded wildfires")
    out = dict(year=a.year, events=int(len(d)), wildfires=int(d.wild.sum()))

    # is it a recorded wildfire
    y, grp = d.wild.to_numpy(), d.place.to_numpy()
    res = {}
    for k in MINE + THEIRS:
        s = d[k].to_numpy()
        res[k] = boot(lambda i: roc_auc_score(y[i], s[i]), grp, a.boot, rng)
    logit = lambda: LogisticRegression(C=1.0, max_iter=1000)  # noqa: E731
    fits = {"GOES first scan": GOES_X, "GOES + volume": GOES_X + ["vol_in", "vol_around"],
            "GOES + volume + kinds": GOES_X + ["vol_in", "vol_around", "tree_in", "shrub_in", "herb_in"],
            "volume + kinds alone": ["vol_in", "vol_around", "tree_in", "shrub_in", "herb_in"],
            "GOES + their kNN fuel": GOES_X + ["knn_in", "knn_around"]}
    for name, cols in fits.items():
        s = oof(d[cols].to_numpy(float), y, grp, logit)
        res[name] = boot(lambda i: roc_auc_score(y[i], s[i]), grp, a.boot, rng)
    out["wildfire or not, AUC"] = res
    out["medians"] = {lab: {k: round(float(g[k].median() if not k.startswith("vol") else np.exp(g[k].median()) - 0.02), 3)
                            for k in MINE} for lab, g in d.groupby("label")}

    # how big it gets: one row per fire, at its first event
    f = d[(d.wild == 1) & d.fire_acres.notna() & (d.fire_acres > 0)].sort_values("t0").drop_duplicates(["fire", "fire_acres"])
    f = f.reset_index(drop=True)
    la, fg = np.log10(f.fire_acres.to_numpy()), f.place.to_numpy()
    big = (f.fire_acres.to_numpy() >= 100).astype(int)
    size, past = {}, {}
    for k in MINE + THEIRS + GOES_X:
        s = f[k].to_numpy()
        size[k] = boot(lambda i: spearmanr(s[i], la[i]).statistic, fg, a.boot, rng)
        past[k] = boot(lambda i: roc_auc_score(big[i], s[i]), fg, a.boot, rng)
    ridge = lambda: Ridge(alpha=1.0)  # noqa: E731
    for name, cols in fits.items():
        s = oof(f[cols].to_numpy(float), la, fg, ridge)
        size[name] = boot(lambda i: spearmanr(s[i], la[i]).statistic, fg, a.boot, rng)
        s = oof(f[cols].to_numpy(float), big, fg, logit)
        past[name] = boot(lambda i: roc_auc_score(big[i], s[i]), fg, a.boot, rng)
    out["size"] = dict(fires=int(len(f)), past_100_acres=int(big.sum()),
                       acres_quartiles=[round(float(q), 1) for q in np.percentile(f.fire_acres, [25, 50, 75])])
    out["size, Spearman with log acres"] = size
    out["past 100 acres, AUC"] = past

    print(f"\nrecorded wildfire or not ({len(d)} events, {int(y.sum())} wildfires), AUC")
    for k, v in res.items():
        print(f"  {k:26s} {v[0]:.3f} ({v[1]:.2f}-{v[2]:.2f})")
    print("\nmedians by label (volume in m, the rest shares)")
    for lab, v in out["medians"].items():
        print(f"  {lab:10s} {v}")
    print(f"\nsize: {len(f)} fires, {int(big.sum())} past 100 acres, quartiles {out['size']['acres_quartiles']} acres")
    print(f"  {'':26s} {'Spearman, log acres':>22s} {'AUC past 100 acres':>22s}")
    for k in size:
        print(f"  {k:26s} {size[k][0]:>8.3f} ({size[k][1]:.2f}-{size[k][2]:.2f}) {past[k][0]:>8.3f} ({past[k][1]:.2f}-{past[k][2]:.2f})")
    (ROOT / "results" / f"goes_fuel_role_{a.year}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
