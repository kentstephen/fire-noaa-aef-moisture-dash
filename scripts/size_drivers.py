# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]
# ///
"""What explains how big a fire gets, next to fuel? The lower 48, 2020 to 2026, the fires of start_fuel.py.

One row per fire, everything known at the start or in the first 3 days:
  fuel        volume within 1, 5 and 10 km of the origin, share of ground over 2 m (results/start_fuel.json)
  kind        tree, shrub, herb and bare shares within 5 km (same file)
  weather     Fosberg at the hour of discovery, its peak and mean over 72 hours, lowest humidity, highest wind
              (results/start_weather.parquet)
  remoteness  distance to a drivable road, a major road, a track, a power line, a building; buildings within about
              1 km (results/overture_at_points.parquet)
  season      day of the year (as sine and cosine)

Each alone and in sets, ridge fits of log acres scored out of fold two ways: by 1 degree place (5 folds) and by
year (each year held out). Spearman with log acres, and AUC for passing 1,000 and 10,000 acres. Intervals: 95 %,
resampling 1 degree places. The sample has equal numbers per size class up to 10,000 acres and every fire above,
so big fires count for far more here than among all starts; read the numbers as "can it tell big from small",
not as a forecast for an ordinary start.

Under 1 GB of RAM. Writes results/size_drivers.json (a few KB).

    uv run scripts/size_drivers.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SETS = {
    "fuel": ["l_vol_1", "l_vol_5", "l_vol_10", "heavy_10"],
    "kind": ["tree_5", "shrub_5", "herb_5", "none_5"],
    "weather": ["ff_0", "ff_max72", "ff_mean72", "rh_min72", "wind_max72"],
    "remoteness": ["l_km_drivable", "l_km_major", "l_km_track", "l_km_power", "l_km_building", "l_buildings"],
    "season": ["doy_sin", "doy_cos"],
}
COMBOS = {"fuel + kind": ["fuel", "kind"], "fuel + kind + weather": ["fuel", "kind", "weather"],
          "fuel + kind + remoteness": ["fuel", "kind", "remoteness"], "everything but fuel and kind": ["weather", "remoteness", "season"],
          "everything": ["fuel", "kind", "weather", "remoteness", "season"]}


def oof(X, y, groups, splitter):
    p = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups):
        p[te] = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(X[tr], y[tr]).predict(X[te])
    return p


def score(s, la, acres, place, rng, n=300):
    ug = np.unique(place)
    idx = {g: np.nonzero(place == g)[0] for g in ug}
    fns = {"spearman": lambda i: spearmanr(s[i], la[i]).statistic,
           "auc_1k": lambda i: roc_auc_score(acres[i] >= 1000, s[i]), "auc_10k": lambda i: roc_auc_score(acres[i] >= 10000, s[i])}
    draws = [np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]) for _ in range(n)]
    out = {}
    for k, fn in fns.items():
        b = [fn(i) for i in draws]
        out[k] = [round(float(fn(np.arange(len(s)))), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    return out


def main():
    rng = np.random.default_rng(0)
    d = pd.DataFrame(json.load(open(ROOT / "results" / "start_fuel.json"))["rows"])
    w = pd.read_parquet(ROOT / "results" / "start_weather.parquet")
    o = pd.read_parquet(ROOT / "results" / "overture_at_points.parquet")
    o = o[o.set == "fires"].rename(columns={"key": "irwin"}).drop(columns=["set", "lon", "lat"]).drop_duplicates("irwin")
    n0 = len(d)
    d = d.merge(w, on="irwin").merge(o, on="irwin")
    for km in (1, 5, 10):
        d[f"l_vol_{km}"] = np.log(0.02 + d[f"vol_{km}"])
    for k in ("drivable", "major", "track", "power", "building"):
        d[f"l_km_{k}"] = np.log(0.125 + d[f"km_{k}"])
    d["l_buildings"] = np.log1p(d.buildings_1km)
    doy = pd.to_datetime(d.discovered, utc=True).dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    cols = sorted({c for v in SETS.values() for c in v})
    d = d.dropna(subset=cols).reset_index(drop=True)
    la, acres = np.log10(d.acres.to_numpy()), d.acres.to_numpy()
    place = (np.floor(d.lon) * 1000 + np.floor(d.lat)).to_numpy()
    print(f"{len(d)} of {n0} fires with fuel, weather and Overture readings; {int((acres >= 1000).sum())} past 1,000 acres, "
          f"{int((acres >= 10000).sum())} past 10,000")
    out = dict(fires=int(len(d)), single={}, by_place={}, by_year={})
    print("\none reading at a time (sign shows the direction: + means more of it, bigger fire)")
    for c in cols:
        out["single"][c] = score(d[c].to_numpy(float), la, acres, place, rng)
        v = out["single"][c]
        print(f"  {c:16s} Spearman {v['spearman'][0]:+.3f} ({v['spearman'][1]:+.2f} to {v['spearman'][2]:+.2f})   "
              f"AUC past 1k {v['auc_1k'][0]:.3f}   past 10k {v['auc_10k'][0]:.3f}")
    fits = {**{k: [k] for k in SETS}, **COMBOS}
    for how, split, grp in (("by_place", GroupKFold(5), place), ("by_year", LeaveOneGroupOut(), d.year.to_numpy())):
        print(f"\nfits scored out of fold, {how.replace('_', ' ')}")
        for name, sets in fits.items():
            X = d[[c for s in sets for c in SETS[s]]].to_numpy(float)
            out[how][name] = score(oof(X, la, grp, split), la, acres, place, rng)
            v = out[how][name]
            print(f"  {name:30s} Spearman {v['spearman'][0]:+.3f} ({v['spearman'][1]:+.2f} to {v['spearman'][2]:+.2f})   "
                  f"AUC past 1k {v['auc_1k'][0]:.3f} ({v['auc_1k'][1]:.2f} to {v['auc_1k'][2]:.2f})   "
                  f"past 10k {v['auc_10k'][0]:.3f} ({v['auc_10k'][1]:.2f} to {v['auc_10k'][2]:.2f})")
    (ROOT / "results" / "size_drivers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
