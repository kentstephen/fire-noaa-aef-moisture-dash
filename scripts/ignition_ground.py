# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Does the ground say where fires start? The lower 48, 2019 to 2026, fuel volume next to a score fit on starts.

Cases and controls: aef-landcover-fosberg-cog/data/ignition (cases_big.parquet, embeddings_big.parquet,
nlcd_big.parquet): 22,209 WFIGS wildfire starts with a known cause, each with a point up to 2.5 km away, one 5 to
10 km away and one 20 to 30 km away, and the 10 m AlphaEarth pixel of the year before at all four. Nothing is read
from the network.

At every point, from that one pixel:
  volume      the national volume head (models/fuel_volume_conus.json), cover x height in m: what the map draws
  kind        the head's leading lifeform: none, tree, shrub, herb
  start head  a logistic head on the 64 embedding values, fit on starts against the 20 to 30 km points and
              scored out of fold (5 folds by 135 km weather square, so nearby fires never sit on both sides)
  both        the same with volume and kind added to the 64 values
When results/overture_at_points.parquet exists (overture_points.py), also the distance to the nearest drivable
road, major road, track, power line and building, the buildings within about 1 km, and heads fit on those.

AUC: the chance a start outscores its control (0.5 = coin flip). Intervals: 95 %, resampling weather squares.
Top share: the share of starts on ground that scores in the top 10 % and 25 % of the 20 to 30 km points (a score
that knows nothing puts 10 % and 25 % there).

Under 2 GB of RAM. Writes results/ignition_ground.json (a few KB).

    uv run scripts/ignition_ground.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("FUEL_HEAD", "fuel_volume_conus.json")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_fuel as bf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT.parent / "aef-landcover-fosberg-cog" / "data" / "ignition"
RINGS = {"up to 2.5 km": "s_ctrl", "5 to 10 km": "s10_ctrl", "20 to 30 km": "s30_ctrl"}
REGIONS = {"Pacific": ["CA", "OR", "WA"], "Mountain": ["ID", "MT", "WY", "NV", "UT", "CO"],
           "Southwest": ["AZ", "NM", "TX", "OK"], "Plains": ["ND", "SD", "NE", "KS", "MN", "IA", "MO", "WI", "IL"],
           "Southeast": ["FL", "GA", "AL", "MS", "LA", "AR", "TN", "KY", "SC", "NC", "VA", "WV"]}
REGION_OF = {f"US-{st}": r for r, sts in REGIONS.items() for st in sts}
KINDS = ["none", "tree", "shrub", "herb"]
OVERTURE = ["drivable", "major", "track", "power", "building"]
BOOT = 300


def auc(pos, neg):
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auc(d, score, ctrl, rng):
    """AUC of cases against one control kind, with an interval from resampling weather squares."""
    c, k = d[d.kind == "case"], d[d.kind == ctrl]
    bc = {g: v[score].to_numpy() for g, v in c.groupby("sq")}
    bk = {g: v[score].to_numpy() for g, v in k.groupby("sq")}
    sqs = np.array([g for g in bc if g in bk])
    if len(sqs) < 5:
        return None
    out = []
    for _ in range(BOOT):
        draw = rng.choice(sqs, len(sqs))
        out.append(auc(np.concatenate([bc[g] for g in draw]), np.concatenate([bk[g] for g in draw])))
    return [round(auc(c[score].to_numpy(), k[score].to_numpy()), 3), round(float(np.percentile(out, 2.5)), 3),
            round(float(np.percentile(out, 97.5)), 3)]


def main():
    rng = np.random.default_rng(0)
    c = pd.read_parquet(D / "cases_big.parquet")
    e = pd.read_parquet(D / "embeddings_big.parquet")
    n = pd.read_parquet(D / "nlcd_big.parquet")
    if "row" in e.columns:
        e = e.set_index("row")
    if "row" in n.columns:
        n = n.set_index("row")
    d = c.join(e[["pix", "nodata"]], how="inner").join(n[["nlcd"]], how="left")
    d = d[~d.nodata & d.kind.isin(["case", *RINGS.values()])].copy()
    d["row"] = d.index.astype(str)
    ov = ROOT / "results" / "overture_at_points.parquet"
    have_ov = ov.exists()
    if have_ov:  # distances to roads, power lines and buildings (overture_points.py)
        o = pd.read_parquet(ov)
        o = o[o.set == "starts"].drop(columns=["set", "lon", "lat"]).set_index("key")
        d = d.join(o, on="row", how="inner")
        for k in OVERTURE:
            d[f"near {k}"] = -d[f"km_{k}"]
        d["buildings"] = np.log1p(d.buildings_1km)
    q = np.stack([np.frombuffer(b, np.int8) for b in d.pix])
    vol, life = bf.volume(q)
    f = q.astype(np.float32) / 127.5
    P = np.sign(f) * f * f
    d["volume"] = np.log(0.02 + vol)
    d["vol_m"] = vol
    d["life"] = life
    if have_ov:
        out_median = {k: {kind: round(float(d[d.kind == kind][f"km_{k}"].median()), 2) for kind in ["case", *RINGS.values()]} for k in OVERTURE}
        print("median km to the nearest, starts and their controls:", out_median, flush=True)
    for i, k in enumerate(KINDS):
        d[k] = (life == i).astype(float)
    d["vegetated"] = (life > 0).astype(float)
    d["sq"] = d.cy * 100 + d.cx
    d["region"] = d.state.map(REGION_OF).fillna("Northeast")
    d["water"] = d.nlcd == 11
    # a case counts only with all three of its controls read
    full = d.groupby("case_id").kind.nunique()
    keep = d.case_id.isin(full[full == 4].index).to_numpy()
    d, P = d[keep].reset_index(drop=True), P[keep]
    print(f"{int((d.kind == 'case').sum())} starts with all three controls, {d.year.min()} to {d.year.max()}; "
          f"{int((d.kind == 'case').sum() and (d[d.kind == 'case'].cause == 'Human').sum())} human, "
          f"{int((d[d.kind == 'case'].cause == 'Natural').sum())} natural", flush=True)

    # heads fit on starts against the far ring, scored out of fold on every point of the held out squares
    far = d.kind.isin(["case", "s30_ctrl"]).to_numpy()
    y = (d.kind == "case").to_numpy(int)
    extra = d[["volume", "tree", "shrub", "herb"]].to_numpy(np.float32)
    sets = {"start head": P, "both": np.hstack([P, extra]), "volume + kind": extra}
    if have_ov:
        O = np.hstack([np.log(0.125 + d[[f"km_{k}" for k in OVERTURE]].to_numpy(np.float32)), d[["buildings"]].to_numpy(np.float32)])
        sets.update({"overture": O, "overture + volume + kind": np.hstack([O, extra]), "start head + overture": np.hstack([P, O])})
    for name in sets:
        d[name] = np.nan
    for tr, te in GroupKFold(5).split(P, groups=d.sq):
        tr = tr[far[tr]]
        for name, X in sets.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[tr], y[tr])
            d.loc[d.index[te], name] = m.decision_function(X[te])
    scores = ["volume", "tree", "shrub", "herb", "vegetated", "volume + kind", "start head", "both"]
    tops = ["volume", "volume + kind", "start head", "both"]
    if have_ov:
        scores += [f"near {k}" for k in OVERTURE] + ["buildings", "overture", "overture + volume + kind", "start head + overture"]
        tops += ["near drivable", "overture", "start head + overture"]

    out = dict(starts=int(y.sum()), years=[int(d.year.min()), int(d.year.max())], auc={}, top={}, medians={}, regions={})
    groups = {"all starts": np.ones(len(d), bool), "human": (d.cause == "Human").to_numpy(),
              "natural": (d.cause == "Natural").to_numpy()}
    for gname, sel in groups.items():
        g = d[sel]
        cases = g[g.kind == "case"]
        out["auc"][gname] = {ring: {s: boot_auc(g, s, kind, rng) for s in scores} for ring, kind in RINGS.items()}
        dry = g[~g.case_id.isin(g[g.water].case_id)]  # starts whose controls are all on land
        out["auc"][gname]["20 to 30 km, no open water"] = {s: boot_auc(dry, s, "s30_ctrl", rng) for s in scores}
        farc = g[g.kind == "s30_ctrl"]
        out["top"][gname] = {s: [round(float((cases[s] > farc[s].quantile(1 - p)).mean()), 3) for p in (0.10, 0.25)]
                             for s in tops}
        out["medians"][gname] = {
            name: dict(n=int(len(v)), volume_m=round(float(v.vol_m.median()), 2), volume_mean_m=round(float(v.vol_m.mean()), 2),
                       **{k: round(float(v[k].mean()), 3) for k in KINDS})
            for name, v in (("starts", cases), *((ring, g[g.kind == kind]) for ring, kind in RINGS.items()))}
        print(f"\n{gname}: {len(cases)} starts")
        for name, v in out["medians"][gname].items():
            print(f"  {name:14s} {v}")
        for part in (scores[:8], scores[8:]):
            if part:
                print(f"  {'AUC against':28s}" + "".join(f"{s:>{max(15, len(s) + 2)}s}" for s in part))
                for ring, res in out["auc"][gname].items():
                    print(f"  {ring:28s}" + "".join(f"{res[s][0]:>{max(15, len(s) + 2)}.3f}" for s in part))
        print("  share of starts on the top 10 % / 25 % of far ground: "
              + ", ".join(f"{s} {v[0]:.0%} / {v[1]:.0%}" for s, v in out["top"][gname].items()))
    print("\nby region, against 20 to 30 km (all causes)")
    print(f"  {'':12s}{'starts':>8s}" + "".join(f"{s:>15s}" for s in ("volume", "herb", "start head", "both")))
    for reg, g in d.groupby("region"):
        res = {s: boot_auc(g, s, "s30_ctrl", rng) for s in ("volume", "herb", "start head", "both")}
        out["regions"][reg] = dict(starts=int((g.kind == "case").sum()), **res)
        print(f"  {reg:12s}{out['regions'][reg]['starts']:>8d}" + "".join(f"{res[s][0]:>15.3f}" for s in res))
    print("\nby year, against 20 to 30 km (all causes)")
    out["years_auc"] = {}
    for yr, g in d.groupby("year"):
        res = {s: boot_auc(g, s, "s30_ctrl", rng) for s in ("volume", "start head")}
        out["years_auc"][int(yr)] = res
        print(f"  {yr}  volume {res['volume'][0]:.3f}  start head {res['start head'][0]:.3f}")
    (ROOT / "results" / "ignition_ground.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
