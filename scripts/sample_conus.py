# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3", "obstore", "duckdb>=2.0.1", "shapely", "rasterio", "psutil", "pyarrow"]
# ///
"""Training pixels for a volume head that works across CONUS: embeddings and LANDFIRE labels in windows spread
evenly over the lower 48 states.

The California head (models/fuel_volume.json) was fit on 728 windows inside California (aef-hls-s1-fire-knn,
scripts/sample_change.py). This draws the rest of the country the same way so the two sets can be fit together:

Windows: mosaic chunks (256 x 256 at 10 m), --per of them at random in every 1 degree cell whose chunk centers
fall inside the lower 48 (Census 1:500k outline), California left out because it is already sampled.

Per window --px random pixels with an embedding in each of the years LANDFIRE's editions are paired with
(LF2016 with 2017, then 2022, 2024, 2025), and LANDFIRE's canopy cover (CC), canopy height (CH), fuel model
(FBFM40) and existing vegetation cover and height (EVC, EVH) from each edition at those pixels.

Writes data/conus/w_<cy>_<cx>.npz (q: int8 [n, 4, 64] in YEARS order; rows, cols; cc_, ch_, fb_, evc_, evh_<ed>:
int16 [n]) and data/conus/windows.parquet (cy, cx, lon, lat). Resumable: windows already written are skipped.

RAM: one window in flight is 4 x 4 MB of embeddings and 20 small rasters, under 2 GB with 6 workers; stops past
--max-gb.

    uv run scripts/sample_conus.py [--per 4] [--px 1500] [--workers 6] [--limit N]
"""
import argparse
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "conus"
RES, X0, Y0, CHUNK = 8.983111749910169e-05, -180.0, 83.68570533713473, 256
EDITIONS = [2016, 2022, 2024, 2025]
YEARS = [2017, 2022, 2024, 2025]  # the embedding year each edition is paired with
LAYERS = {"cc": "CC", "ch": "CH", "fb": "FBFM40", "evc": "EVC", "evh": "EVH"}
LF_URL = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF{v}/LF{v}_{lyr}_CONUS/ImageServer/exportImage"
STATES = "/vsizip//vsicurl/https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"


def outlines():
    """The lower 48 and California as shapely geometries; the lower 48 is fetched once and kept as WKB."""
    import duckdb
    import shapely
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    path = ROOT / "data" / "conus_500k.parquet"
    if not path.exists():
        con.execute(f"""COPY (SELECT ST_AsWKB(ST_Union_Agg(geom)) AS wkb FROM ST_Read('{STATES}')
                        WHERE STUSPS NOT IN ('AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS')) TO '{path}'""")
    conus = shapely.from_wkb(bytes(con.execute(f"SELECT wkb FROM '{path}'").fetchone()[0]))
    ca = shapely.from_wkb(bytes(con.execute(f"SELECT wkb FROM '{ROOT}/data/ca_500k.parquet'").fetchone()[0]))
    return conus, ca


def pick(per, seed=0):
    """Chunks (cy, cx, lon, lat of the center): per random ones in each 1 degree cell, inside the lower 48, outside California."""
    import shapely
    conus, ca = outlines()
    shapely.prepare(conus)
    shapely.prepare(ca)
    rng = np.random.default_rng(seed)
    w0, s0, e0, n0 = conus.bounds
    out, seen = [], set()
    for lat0 in range(int(np.floor(s0)), int(np.ceil(n0))):
        for lon0 in range(int(np.floor(w0)), int(np.ceil(e0))):
            got = 0
            for _ in range(per * 6):
                lon, lat = lon0 + rng.random(), lat0 + rng.random()
                cy, cx = int((Y0 - lat) / RES) // CHUNK, int((lon - X0) / RES) // CHUNK
                clon, clat = X0 + (cx + 0.5) * CHUNK * RES, Y0 - (cy + 0.5) * CHUNK * RES
                if (cy, cx) in seen or not shapely.contains_xy(conus, clon, clat) or shapely.contains_xy(ca, clon, clat):
                    continue
                seen.add((cy, cx))
                out.append((cy, cx, clon, clat))
                got += 1
                if got == per:
                    break
    return out


def get(url, params, tries=5):
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(q, timeout=180) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 + 4 * i)


def landfire(cy, cx, ed, lyr):
    from rasterio.io import MemoryFile
    lon0, lat0 = X0 + cx * CHUNK * RES, Y0 - cy * CHUNK * RES
    lon1, lat1 = lon0 + CHUNK * RES, lat0 - CHUNK * RES
    b = get(LF_URL.format(v=ed, lyr=lyr), dict(bbox=f"{lon0},{lat1},{lon1},{lat0}", bboxSR=4326, imageSR=4326,
                                               size=f"{CHUNK},{CHUNK}", format="tiff", pixelType="S16",
                                               interpolation="RSP_NearestNeighbor", f="image"))
    with MemoryFile(b) as m, m.open() as r:
        return r.read(1).astype(np.int16)


def mosaic():
    import zarr
    from obstore.store import S3Store
    from zarr.storage import ObjectStore
    zarr.config.set({"async.concurrency": 16})
    store = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                    virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic")
    g = zarr.open_group(ObjectStore(store, read_only=True), mode="r")
    return g["embeddings"], [int(t) for t in g["time"][:]]


def one(arr, times, cy, cx, px, seed):
    path = OUT / f"w_{cy}_{cx}.npz"
    if path.exists():
        return 0
    for attempt in range(5):
        try:
            q = np.stack([np.asarray(arr[times.index(y), :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK]) for y in YEARS])
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 + 5 * attempt)
    q = q.reshape(len(YEARS), 64, -1)
    ok = np.flatnonzero(~(q == -128).all(axis=1).any(axis=0))
    tmp = path.with_suffix(".tmp.npz")
    if len(ok) < 200:
        np.savez_compressed(tmp, q=np.zeros((0, len(YEARS), 64), np.int8))
        tmp.rename(path)
        return 0
    sel = np.sort(np.random.default_rng(seed).choice(ok, min(px, len(ok)), replace=False))
    out = dict(q=np.ascontiguousarray(q[:, :, sel].transpose(2, 0, 1)), rows=(sel // CHUNK).astype(np.int16),
               cols=(sel % CHUNK).astype(np.int16))
    del q
    jobs = [(ed, k, lyr) for ed in EDITIONS for k, lyr in LAYERS.items()]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (ed, k, _), g in zip(jobs, ex.map(lambda j: landfire(cy, cx, j[0], j[2]), jobs)):
            out[f"{k}_{ed}"] = g.ravel()[sel]
    np.savez_compressed(tmp, **out)
    tmp.rename(path)
    return len(sel)


def main():
    import psutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--px", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-gb", type=float, default=4.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    w = pick(a.per)
    pq.write_table(pa.table({"cy": [r[0] for r in w], "cx": [r[1] for r in w], "lon": [r[2] for r in w], "lat": [r[3] for r in w]}),
                   OUT / "windows.parquet")
    todo = [(i, r) for i, r in enumerate(w) if not (OUT / f"w_{r[0]}_{r[1]}.npz").exists()]
    if a.limit:
        todo = todo[::max(1, len(todo) // a.limit)][:a.limit]
    print(f"{len(w)} windows, {len(todo)} to read", flush=True)
    arr, times = mosaic()
    proc = psutil.Process(os.getpid())
    peak, t0, failed = 0.0, time.time(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, arr, times, r[0], r[1], a.px, i) for i, r in todo]
        for i, f in enumerate(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED {todo[i][1][0]}_{todo[i][1][1]}: {e}", flush=True)
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}; rerun to resume", flush=True)
                os._exit(2)
            if i % 25 == 0 or i == len(futs) - 1:
                el = time.time() - t0
                print(f"{i + 1}/{len(futs)}  {el / (i + 1):.1f} s/window  eta {el / (i + 1) * (len(futs) - i - 1) / 60:.0f} min  "
                      f"rss {gb:.2f} GB (peak {peak:.2f})  failed {failed}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB, failed {failed}", flush=True)


if __name__ == "__main__":
    main()
