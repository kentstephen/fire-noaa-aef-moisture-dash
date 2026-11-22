# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3", "obstore", "duckdb>=2.0.1", "psutil", "shapely", "pyarrow"]
# ///
"""The fuel bake carried past the lower 48: Mexico or Canada, with the national head.

The same work as bake_fuel.py --area conus (same windows of 1024 pixels, same A5 resolution, same head,
models/fuel_volume_conus.json), over one more country. The head was fit on LANDFIRE, which stops at the US border,
so over boreal forest, tundra and the tropics it is reading ground it never saw: the cells are drawn, not checked.

Windows the lower 48 bake already covers (it reads whole windows along the border) are left out. Parts go to
data/parts_<area>/<row>_<col>.parquet, south to north, and a stopped run picks up where it left off.

The outline is Natural Earth's 1:50m countries (public domain), read once into data/ne_50m_countries.geojson
(3 MB) and widened by 0.1 degrees so coarse coastlines do not drop a coastal window.

Windows: Mexico 21,521 and Canada 209,219 before the border windows are taken out. RAM as the national bake:
6.9 GB measured across 10 workers; each worker stops itself over --max-gb.

    uv run scripts/bake_neighbors.py --area mexico [--workers 10] [--res 17] [--limit N]
    uv run scripts/bake_neighbors.py --area canada
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from multiprocessing import Pool
from pathlib import Path

AREAS = {"mexico": "Mexico", "canada": "Canada"}
_area = next((sys.argv[i + 1] for i, v in enumerate(sys.argv[:-1]) if v == "--area"), os.environ.get("FUEL_AREA", "mexico"))
os.environ.update(FUEL_AREA=_area, FUEL_PARTS=f"parts_{_area}", FUEL_HEAD="fuel_volume_conus.json")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_fuel as bf  # noqa: E402  (reads the parts folder and the head from the environment set above)

ROOT = Path(__file__).resolve().parent.parent
OUTLINE = ROOT / "data" / "ne_50m_countries.geojson"
SOURCE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"


def windows(area, win):
    import numpy as np
    import shapely
    if not OUTLINE.exists():
        urllib.request.urlretrieve(SOURCE, OUTLINE)
    fc = json.load(open(OUTLINE))
    g = next(shapely.geometry.shape(f["geometry"]) for f in fc["features"] if f["properties"]["ADMIN"] == AREAS[area])
    g = g.buffer(0.1)
    shapely.prepare(g)
    lon0, lat0, lon1, lat1 = g.bounds
    lat1 = min(lat1, bf.Y0)
    c0, c1 = int((lon0 - bf.X0) / bf.RES_DEG) // win, int((lon1 - bf.X0) / bf.RES_DEG) // win
    r0, r1 = int((bf.Y0 - lat1) / bf.RES_DEG) // win, int((bf.Y0 - lat0) / bf.RES_DEG) // win
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    d = win * bf.RES_DEG
    hit = shapely.intersects(g, shapely.box(bf.X0 + cc * d, bf.Y0 - (rr + 1) * d, bf.X0 + (cc + 1) * d, bf.Y0 - rr * d))
    done = set(bf.windows(win, "conus"))
    out = [(int(r), int(c)) for r, c in zip(rr[hit], cc[hit]) if (int(r), int(c)) not in done]
    return sorted(out, key=lambda rc: (-rc[0], rc[1]))  # south to north


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", choices=list(AREAS), default="mexico")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--res", type=int, default=17)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-gb", type=float, default=3.0)
    a = ap.parse_args()
    bf.PARTS.mkdir(parents=True, exist_ok=True)
    jobs = [(r, c, a.win, a.res, a.year, a.max_gb) for r, c in windows(a.area, a.win)]
    total = len(jobs)
    jobs = [j for j in jobs if not (bf.PARTS / f"{j[0]}_{j[1]}.parquet").exists()]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{a.area}: {total} windows, {len(jobs)} to read, parts in {bf.PARTS}", flush=True)
    t0, px, done = time.time(), 0, 0
    with Pool(a.workers, maxtasksperchild=300) as pool:
        for n, _dt in pool.imap_unordered(bf.one, jobs):
            done += 1
            px += n
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"{done}/{len(jobs)} windows  {px / 1e6:.0f} M pixels  {el / 60:.1f} min  "
                      f"{done * a.win * a.win * 64 / 1e6 / el:.0f} MB/s decoded", flush=True)


if __name__ == "__main__":
    main()
