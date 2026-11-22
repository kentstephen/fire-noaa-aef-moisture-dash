# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "duckdb>=2.0.1", "pyproj", "scipy", "psutil"]
# ///
"""How far each fire start (and each control point) is from roads, power lines and buildings. The lower 48.

Overture Maps, latest release, read from its public bucket by longitude band; nothing of it is kept on disk.
  roads       transportation segments of subtype road, three groups: major (motorway, trunk, primary, secondary),
              drivable (those plus tertiary, residential, unclassified, service, living_street), and tracks and paths
  power       base / infrastructure lines of class power_line and minor_line
  buildings   the corner of each building's box (no shapes are read)
Every vertex of a line, and a point every 0.002 degrees along it, marks a cell of a 250 m grid (Albers, EPSG:5070).
Distance is then the distance to the nearest marked cell, so it is good to about 250 m. Buildings within about
1 km is the count in the 9 x 9 cells (2.25 km square) around the point.

Points read:
  starts    the 22,209 WFIGS starts of aef-landcover-fosberg-cog (cases_big.parquet) and their three controls
  fires     the fires of start_fuel.py (the same draw)

RAM: DuckDB is capped at 5 GB and opened fresh for each band; the grid is 18,600 x 11,700 cells, 0.2 GB per
layer as flags and 0.9 GB of building counts. About 10 GB at the peak. Stops past --max-gb.
Writes results/overture_at_points.parquet (about 2 MB).

    uv run scripts/overture_points.py
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT.parent / "aef-landcover-fosberg-cog" / "data" / "ignition" / "cases_big.parquet"
BUCKET = "s3://overturemaps-us-west-2/release"
BOX = (-125.0, 24.0, -66.5, 49.6)
X0, Y0, D, NX, NY = -2380000.0, 3200000.0, 250.0, 18600, 11700
STEP = 0.002
GROUPS = {"major": "class IN ('motorway', 'trunk', 'primary', 'secondary')",
          "drivable": "class IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'service', 'living_street')",
          "track": "class IN ('track', 'path')"}
BAND = 3.0


def mark(grid, lon, lat, tr, counts=None):
    x, y = tr.transform(lon, lat)
    c, r = ((x - X0) / D).astype(np.int64), ((Y0 - y) / D).astype(np.int64)
    ok = (c >= 0) & (c < NX) & (r >= 0) & (r < NY)
    if counts is None:
        grid[r[ok], c[ok]] = True
    else:
        np.add.at(grid, (r[ok], c[ok]), counts[ok])


def nearest(flag, row, col, cap=800):
    """Distance in km from each point to the nearest marked cell, looking in a square that grows until the
    nearest cell found is closer than the square's edge (so nothing outside it can be nearer). 200 km at most."""
    out = np.full(len(row), cap * D / 1000, np.float32)
    for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())):
        for h in (6, 24, 96, 400, cap):
            r0, c0 = max(0, r - h), max(0, c - h)
            rr, cc = np.nonzero(flag[r0:r + h + 1, c0:c + h + 1])
            if len(rr):
                d = np.sqrt(((rr + r0 - r) ** 2 + (cc + c0 - c) ** 2).min())
                if d <= h or h == cap:
                    out[i] = min(d, cap) * D / 1000
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=16.0)
    ap.add_argument("--bands", default="")  # e.g. "-121,-118" to try a slice
    a = ap.parse_args()
    import duckdb
    from pyproj import Transformer
    from scipy import ndimage
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    rel = json.load(urllib.request.urlopen("https://stac.overturemaps.org/catalog.json"))["latest"]

    def connect():
        con = duckdb.connect()
        for q in ("INSTALL spatial", "LOAD spatial", "INSTALL httpfs", "LOAD httpfs", "SET geometry_always_xy = true",
                  "SET memory_limit = '5GB'", "SET preserve_insertion_order = false"):
            con.execute(q)
        return con
    proc, t0 = psutil.Process(os.getpid()), time.time()

    def check(what):
        gb = proc.memory_info().rss / 1e9
        print(f"{what}  {(time.time() - t0) / 60:.1f} min  {gb:.1f} GB", flush=True)
        if gb > a.max_gb:
            print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}", flush=True)
            os._exit(2)

    lo, hi = (map(float, a.bands.split(",")) if a.bands else (BOX[0], BOX[2]))
    bands = [(x, min(x + BAND, hi)) for x in np.arange(lo, hi, BAND)]
    flags = {k: np.zeros((NY, NX), bool) for k in (*GROUPS, "power", "building")}
    nbld = np.zeros((NY, NX), np.float32)
    seg = f"read_parquet('{BUCKET}/{rel}/theme=transportation/type=segment/*.parquet')"
    inf = f"read_parquet('{BUCKET}/{rel}/theme=base/type=infrastructure/*.parquet')"
    bld = f"read_parquet('{BUCKET}/{rel}/theme=buildings/type=building/*.parquet')"
    # every vertex, and for lines longer than a cell a point every {s} degrees along them
    cells = """WITH src AS ({src}), pt AS (
                   SELECT g, unnest(ST_Dump(ST_Points(geometry))).geom AS p FROM src
                   UNION ALL
                   SELECT g, unnest(ST_Dump(ST_LineInterpolatePoints(geometry, {s} / ST_Length(geometry), true))).geom AS p
                   FROM src WHERE ST_GeometryType(geometry) = 'LINESTRING' AND ST_Length(geometry) > {s})
               SELECT DISTINCT g, floor(ST_X(p) / {s})::INTEGER AS ix, floor(ST_Y(p) / {s})::INTEGER AS iy FROM pt"""
    for b0, b1 in bands:
        con = connect()  # a new one per band: one kept for the whole run grew by 0.6 GB a band
        where = f"bbox.xmin >= {b0} AND bbox.xmin < {b1} AND bbox.ymin BETWEEN {BOX[1]} AND {BOX[3]}"
        case = "CASE " + " ".join(f"WHEN {w} THEN '{k}'" for k, w in GROUPS.items() if k != "drivable") + " ELSE 'other' END"
        src = f"SELECT {case} AS g, geometry FROM {seg} WHERE subtype = 'road' AND ({GROUPS['drivable']} OR {GROUPS['track']}) AND {where}"
        got = con.execute(cells.format(s=STEP, src=src)).fetchnumpy()
        lon, lat = (got["ix"] + 0.5) * STEP, (got["iy"] + 0.5) * STEP
        for k in ("major", "track"):
            m = got["g"] == k
            mark(flags[k], lon[m], lat[m], tr)
        m = got["g"] != "track"
        mark(flags["drivable"], lon[m], lat[m], tr)
        src = f"SELECT 'power' AS g, geometry FROM {inf} WHERE class IN ('power_line', 'minor_line') AND {where}"
        got = con.execute(cells.format(s=STEP, src=src)).fetchnumpy()
        mark(flags["power"], (got["ix"] + 0.5) * STEP, (got["iy"] + 0.5) * STEP, tr)
        got = con.execute(f"""SELECT floor(bbox.xmin / {STEP})::INTEGER AS ix, floor(bbox.ymin / {STEP})::INTEGER AS iy, count(*) AS n
                              FROM {bld} WHERE {where} GROUP BY 1, 2""").fetchnumpy()
        mark(nbld, (got["ix"] + 0.5) * STEP, (got["iy"] + 0.5) * STEP, tr, counts=got["n"].astype(np.float32))
        con.close()
        del got
        check(f"band {b0:.0f} to {b1:.0f}: {int(flags['drivable'].sum()):,} road cells, {int(nbld.sum()):,} buildings")
    flags["building"] = nbld > 0

    # the points
    c = pd.read_parquet(CASES, columns=["case_id", "kind", "lon", "lat"])
    c = c[c.kind.isin(["case", "s_ctrl", "s10_ctrl", "s30_ctrl"])]
    pts = [pd.DataFrame(dict(set="starts", key=c.index.astype(str), lon=c.lon.to_numpy(), lat=c.lat.to_numpy()))]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import start_fuel
    f = start_fuel.fires(400)
    pts.append(pd.DataFrame(dict(set="fires", key=f.irwin, lon=f.lon, lat=f.lat)))
    pts = pd.concat(pts, ignore_index=True)
    x, y = tr.transform(pts.lon.to_numpy(), pts.lat.to_numpy())
    col, row = ((x - X0) / D).astype(np.int64), ((Y0 - y) / D).astype(np.int64)
    ok = (col >= 0) & (col < NX) & (row >= 0) & (row < NY)
    if a.bands:  # a slice was read: keep only the points inside it
        ok &= (pts.lon.to_numpy() >= lo + 0.5) & (pts.lon.to_numpy() < hi - 0.5)
    pts, col, row = pts[ok].reset_index(drop=True), col[ok], row[ok]
    for k, flag in flags.items():
        pts[f"km_{k}"] = np.round(nearest(flag, row, col), 2)
        check(f"distance to {k}")
    near = ndimage.uniform_filter(nbld, size=9, mode="constant") * 81
    pts["buildings_1km"] = np.round(near[row, col]).astype(np.int32)
    out = ROOT / "results" / ("overture_at_points.parquet" if not a.bands else "overture_at_points_trial.parquet")
    pts.to_parquet(out, compression="zstd", index=False)
    print(pts.groupby("set")[[c_ for c_ in pts.columns if c_.startswith("km_")] + ["buildings_1km"]].median().to_string())
    print(f"{len(pts)} points, Overture release {rel}, {out.name} {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
