# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "requests"]
# ///
"""Do fires grow toward the heavier fuel? The lower 48, fires of 1,000 acres and up.

For each fire of start_fuel.py that has a perimeter in NIFC's WFIGS Interagency Perimeters service: the direction
from its point of origin to the center of its final perimeter (the way it grew), next to the fuel of the year
before in 8 compass sectors 2 to 10 km around the origin (results/start_fuel.json).

  toward vs away     share of fires with more volume in the sector it grew into than in the opposite one
  toward vs the rest share with more volume in that sector than the mean of the other seven
  rank               where that sector stands among the eight, 1 = heaviest (4.5 if fuel had no say)
  alignment          cosine between the way it grew and the direction of the heaviest fuel (0 if no say)

Only fires whose center sits 1 km or more from the origin (otherwise there is no direction to speak of). Only the
perimeter's center is asked for, not its shape. Intervals: 95 %, resampling fires.
Writes results/growth_direction.json (a few KB).

    uv run scripts/growth_direction.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"
CLASSES = ["1k to 10k", "10k to 100k", "100k and up"]


def centers():
    rows, offset = [], 0
    while True:
        r = requests.get(URL, params=dict(where="poly_GISAcres >= 1000", outFields="poly_IRWINID,poly_GISAcres,poly_IncidentName",
                                          returnGeometry="false", returnCentroid="true", outSR=4326, f="json",
                                          resultOffset=offset, resultRecordCount=1000, orderByFields="OBJECTID"), timeout=120).json()
        for f in r.get("features", []):
            c = f.get("centroid")
            if c and f["attributes"].get("poly_IRWINID"):
                rows.append(dict(irwin=f["attributes"]["poly_IRWINID"].strip("{}").lower(), gis_acres=f["attributes"]["poly_GISAcres"],
                                 clon=c["x"], clat=c["y"]))
        if not r.get("exceededTransferLimit") and len(r.get("features", [])) < 1000:
            break
        offset += len(r["features"])
    d = pd.DataFrame(rows)
    return d.sort_values("gis_acres", ascending=False).drop_duplicates("irwin")


def ci(v, rng, fn=np.mean):
    v = np.asarray(v, float)
    b = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(1000)]
    return [round(float(fn(v)), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def main():
    rng = np.random.default_rng(0)
    f = pd.DataFrame(json.load(open(ROOT / "results" / "start_fuel.json"))["rows"])
    f = f[f.acres >= 1000].copy()
    f["irwin"] = f.irwin.str.strip("{}").str.lower()
    c = centers()
    d = f.merge(c, on="irwin")
    kx = 111.2 * np.cos(np.radians(d.lat))
    dx, dy = (d.clon - d.lon) * kx, (d.clat - d.lat) * 111.2
    d["moved_km"] = np.hypot(dx, dy)
    d["bearing"] = np.degrees(np.arctan2(dx, dy)) % 360
    print(f"{len(f)} fires of 1,000 acres and up in the sample, {len(c)} perimeters of that size in the service, {len(d)} matched")
    d = d[(d.moved_km >= 1) & (d.moved_km < 100)]
    sec = np.array([[np.nan if v is None else v for v in s] for s in d.sectors], float)
    d, sec = d[np.isfinite(sec).all(1)].reset_index(drop=True), sec[np.isfinite(sec).all(1)]
    k = (np.floor((d.bearing + 22.5) % 360 / 45)).astype(int).to_numpy()
    n = np.arange(len(d))
    toward, away = sec[n, k], sec[n, (k + 4) % 8]
    rest = (sec.sum(1) - toward) / 7
    d["more_than_away"] = np.where(toward == away, 0.5, toward > away)
    d["more_than_rest"] = (toward > rest).astype(float)
    d["rank"] = (sec > toward[:, None]).sum(1) + 1
    ang = np.radians(np.arange(8) * 45)
    gx, gy = (sec * np.sin(ang)).sum(1), (sec * np.cos(ang)).sum(1)  # where the fuel is heavier, as a direction
    b = np.radians(d.bearing.to_numpy())
    d["alignment"] = (gx * np.sin(b) + gy * np.cos(b)) / np.maximum(np.hypot(gx, gy), 1e-9)
    d["uneven"] = (sec.max(1) - sec.min(1)) / np.maximum(sec.mean(1), 0.05)
    out = dict(matched=int(len(d)), groups={})
    groups = {"all": np.ones(len(d), bool), **{cl: (d.cls == cl).to_numpy() for cl in CLASSES},
              "center 3 km or more from the origin": (d.moved_km >= 3).to_numpy(),
              "fuel uneven around the origin (top half)": (d.uneven >= d.uneven.median()).to_numpy()}
    print(f"{len(d)} with a center 1 km or more from the origin and fuel read in all 8 sectors; median distance {d.moved_km.median():.1f} km\n")
    print(f"  {'':42s}{'fires':>6s}{'toward > away':>22s}{'toward > the rest':>22s}{'rank (4.5 = none)':>22s}{'alignment (0 = none)':>24s}")
    for name, sel in groups.items():
        g = d[sel]
        if len(g) < 10:
            continue
        res = dict(fires=int(len(g)), more_than_away=ci(g.more_than_away, rng), more_than_rest=ci(g.more_than_rest, rng),
                   rank=ci(g["rank"], rng), alignment=ci(g.alignment, rng))
        out["groups"][name] = res
        fmt = lambda v, p=False: (f"{v[0]:.0%} ({v[1]:.0%} to {v[2]:.0%})" if p else f"{v[0]:+.2f} ({v[1]:+.2f} to {v[2]:+.2f})")  # noqa: E731
        print(f"  {name:42s}{len(g):>6d}{fmt(res['more_than_away'], True):>22s}{fmt(res['more_than_rest'], True):>22s}"
              f"{fmt(res['rank']):>22s}{fmt(res['alignment']):>24s}")
    (ROOT / "results" / "growth_direction.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
