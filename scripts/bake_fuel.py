# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zarr>=3", "obstore", "duckdb>=2.0.1", "psutil", "shapely", "pyarrow"]
# ///
"""Fuel per A5 cell over California or the lower 48 (--area conus, national head), from the 10 m AlphaEarth embeddings of one year.

Each 10 m pixel gets the vegetation volume head (models/fuel_volume.json: cover x height of the leading lifeform,
in m) and the A5 cell (resolution --res) its center falls in. Per cell: the sum of volume, the pixel count and the
counts of tree, shrub and herb pixels. The head runs at 10 m and its outputs are averaged; embeddings are never
averaged. Coarser cells are made later as the mean of their children.

The mosaic is read in windows of --win pixels (whole 256 px chunks), each written to data/parts/<row>_<col>.parquet
as soon as it is done, so a stopped run picks up where it left off. merge() joins the parts.

RAM: one window is 64 * win^2 bytes of int8 (67 MB at 1024) and the head runs 262,144 pixels at a time (about
0.3 GB of float32). About 1 GB per worker; each worker stops itself over --max-gb.

    uv run scripts/bake_fuel.py [--workers 4] [--res 15] [--year 2025] [--limit N]
    uv run scripts/bake_fuel.py --merge
    uv run scripts/bake_fuel.py --area conus --res 17 --workers 10     (parts in data/parts_conus)
"""
import argparse
import json
import os
import sys
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_DEG, X0, Y0 = 8.983111749910169e-05, -180.0, 83.68570533713473
# the area's parts folder and head; main() sets these in the environment so the worker processes see them too
PARTS = ROOT / "data" / os.environ.get("FUEL_PARTS", "parts")

H = json.load(open(ROOT / "models" / os.environ.get("FUEL_HEAD", "fuel_volume.json")))
MU, SD = np.array(H["mean"], np.float32), np.array(H["scale"], np.float32)
W1 = np.array(H["w1"], np.float32).reshape(H["hidden"], H["n_in"])
B1 = np.array(H["b1"], np.float32)
W2 = np.array(H["w2"], np.float32).reshape(H["n_out"], H["hidden"])
B2 = np.array(H["b2"], np.float32)
_m = {}


def guard(max_gb):
    """Stop the process on its own when its memory passes max_gb."""
    import psutil

    def run():
        p = psutil.Process()
        while True:
            gb = p.memory_info().rss / 1e9
            if gb > max_gb:
                print(f"STOP: RSS {gb:.1f} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


def mosaic():
    if "arr" not in _m:
        import zarr
        from obstore.store import S3Store
        from zarr.storage import ObjectStore
        zarr.config.set({"async.concurrency": 32})
        store = S3Store("tge-labs", endpoint="https://data.source.coop", region="us-west-2",
                        virtual_hosted_style_request=False, skip_signature=True, prefix="aef-mosaic")
        g = zarr.open_group(ObjectStore(store, read_only=True), mode="r")
        _m["times"], _m["arr"] = [int(t) for t in g["time"][:]], g["embeddings"]
    return _m["arr"], _m["times"]


def db():
    if "db" not in _m:
        import duckdb
        con = duckdb.connect()
        con.execute("SET threads=2")
        # A5 for DuckDB 2.0.1 and later comes from its author's repository
        con.execute("SET allow_extension_repositories = 'allowed'")
        try:
            con.execute("LOAD a5 FROM query_farm")
        except Exception:  # noqa: BLE001
            con.execute("INSTALL a5 FROM query_farm")
            con.execute("LOAD a5 FROM query_farm")
        _m["db"] = con
    return _m["db"]


def volume(q):
    """int8 (n, 64) -> volume in m (float32) and lifeform index (0 none, 1 tree, 2 shrub, 3 herb)."""
    vol = np.empty(len(q), np.float32)
    life = np.empty(len(q), np.int8)
    for i in range(0, len(q), 262144):
        f = q[i:i + 262144].astype(np.float32) / 127.5
        o = np.maximum(((np.sign(f) * f * f - MU) / SD) @ W1.T + B1, 0) @ W2.T + B2
        vol[i:i + 262144] = np.maximum(np.exp(o[:, 0] * (H["vol_hi"] - H["vol_lo"]) + H["vol_lo"]) - 0.02, 0)
        life[i:i + 262144] = o[:, 16:20].argmax(1)
    return vol, life


def windows(win, area="ca"):
    """Windows (row, col) of win pixels, aligned to the mosaic's chunks, that touch the area (California or the lower 48)."""
    import shapely
    if area == "conus":
        import duckdb
        ca = shapely.from_wkb(duckdb.connect().execute(f"SELECT wkb FROM '{ROOT}/data/conus_500k.parquet'").fetchall()[0][0])
    else:
        rings = json.load(open(ROOT / "data" / "ca.json"))
        ca = shapely.union_all([shapely.Polygon(r) for r in rings])
    shapely.prepare(ca)
    lon0, lat0, lon1, lat1 = ca.bounds
    c0, c1 = int((lon0 - X0) / RES_DEG) // win, int((lon1 - X0) / RES_DEG) // win
    r0, r1 = int((Y0 - lat1) / RES_DEG) // win, int((Y0 - lat0) / RES_DEG) // win
    out = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            box = shapely.box(X0 + c * win * RES_DEG, Y0 - (r + 1) * win * RES_DEG,
                              X0 + (c + 1) * win * RES_DEG, Y0 - r * win * RES_DEG)
            if ca.intersects(box):
                out.append((r, c))
    return out


def one(job):
    r, c, win, res, year, max_gb = job
    out = PARTS / f"{r}_{c}.parquet"
    if out.exists():
        return 0, 0.0
    if "guard" not in _m:
        guard(max_gb)
        _m["guard"] = True
    arr, times = mosaic()
    t = time.time()
    y0, x0 = r * win, c * win
    for attempt in range(5):
        try:
            e = np.asarray(arr[times.index(year), :, y0:y0 + win, x0:x0 + win])
            break
        except Exception as err:  # noqa: BLE001
            if attempt == 4:
                print(f"FAILED {r}_{c}: {err}", flush=True)
                return 0, 0.0
            time.sleep(3 + 5 * attempt)
    nbytes = e.nbytes
    q = e.reshape(64, -1).T
    ok = q[:, 0] != -128
    con = db()
    if ok.any():
        vol, life = volume(q[ok])
        lon = np.broadcast_to(X0 + (x0 + np.arange(win) + 0.5) * RES_DEG, (win, win)).ravel()[ok]
        lat = np.broadcast_to((Y0 - (y0 + np.arange(win) + 0.5) * RES_DEG)[:, None], (win, win)).ravel()[ok]
        px = {"lon": lon, "lat": lat, "vol": vol, "life": life}  # noqa: F841 (read by duckdb)
        con.register("px", con.from_arrow(__import__("pyarrow").table(px)))
        src = f"""SELECT a5_lonlat_to_cell(lon, lat, {res}) AS cell, sum(vol)::FLOAT AS vol, count(*)::INTEGER AS n,
                  count(*) FILTER (life = 1)::INTEGER AS tree, count(*) FILTER (life = 2)::INTEGER AS shrub,
                  count(*) FILTER (life = 3)::INTEGER AS herb FROM px GROUP BY 1 ORDER BY 1"""
    else:
        src = "SELECT 0::UBIGINT AS cell, 0::FLOAT AS vol, 0::INTEGER AS n, 0::INTEGER AS tree, 0::INTEGER AS shrub, 0::INTEGER AS herb WHERE false"
    tmp = out.with_suffix(".tmp")
    con.execute(f"COPY ({src}) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.rename(out)
    return int(ok.sum()), time.time() - t


def merge(res):
    import duckdb
    con = duckdb.connect()
    con.execute("SET allow_extension_repositories = 'allowed'")
    con.execute("LOAD a5 FROM query_farm")
    out = ROOT / "data" / f"fuel_ca_r{res}.parquet"
    con.execute(f"""COPY (SELECT cell, sum(vol)::FLOAT AS vol, sum(n)::INTEGER AS n, sum(tree)::INTEGER AS tree,
                    sum(shrub)::INTEGER AS shrub, sum(herb)::INTEGER AS herb
                    FROM '{PARTS}/*.parquet' GROUP BY 1 ORDER BY 1) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    print(con.sql(f"SELECT count(*) cells, sum(n) pixels, sum(vol) / sum(n) mean_vol FROM '{out}'"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--res", type=int, default=15)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--area", choices=["ca", "conus"], default="ca")
    a = ap.parse_args()
    if a.area == "conus" and "FUEL_PARTS" not in os.environ:
        # the lower 48 with the national head, in its own parts folder; start again so this module reads them
        os.environ.update(FUEL_PARTS="parts_conus", FUEL_HEAD="fuel_volume_conus.json")
        os.execv(sys.executable, [sys.executable, *sys.argv])
    if a.merge:
        return merge(a.res)
    PARTS.mkdir(parents=True, exist_ok=True)
    jobs = [(r, c, a.win, a.res, a.year, a.max_gb) for r, c in windows(a.win, a.area)]
    total = len(jobs)
    jobs = [j for j in jobs if not (PARTS / f"{j[0]}_{j[1]}.parquet").exists()]
    if a.limit:
        jobs = jobs[len(jobs) // 2:len(jobs) // 2 + a.limit]
    print(f"{total} windows, {total - len(jobs) if not a.limit else '?'} done, {len(jobs)} to read", flush=True)
    t0, px, done = time.time(), 0, 0
    with Pool(a.workers, maxtasksperchild=300) as pool:
        for n, _dt in pool.imap_unordered(one, jobs):
            done += 1
            px += n
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"{done}/{len(jobs)} windows  {px / 1e6:.0f} M pixels  {el / 60:.1f} min  "
                      f"{done * a.win * a.win * 64 / 1e6 / el:.0f} MB/s decoded", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
