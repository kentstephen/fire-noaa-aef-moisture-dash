# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pandas", "pyarrow", "pyproj", "psutil", "duckdb>=2.0.1",
#                 "obstore", "shapely", "scipy"]
# ///
"""The weather each fire of start_fuel.py started in: plain Fosberg at the hour of discovery and over the 3 days after.

HRRR analysis (dynamical.org Icechunk, 3 km, hourly), the same record and formula as goes_overlap.py. For each
fire, at the HRRR pixel of its point of origin:
  ff_0        Fosberg at the hour it was discovered
  ff_max72    the highest Fosberg of the next 72 hours
  ff_mean72   the mean of those hours
  rh_min72, wind_max72 (m/s), t_0 (C)
The store keeps 90 days x 135 km in one chunk, so fires are grouped by chunk and each chunk is read once. A fire
discovered in the last 3 days of a 90 day chunk gets the hours that are left in it (hours72 says how many).

RAM: one chunk is 2160 hours x 45 x 45 pixels x 4 fields, 70 MB as float32; about 2 GB across 8 readers.
Writes results/start_weather.parquet (under 1 MB).

    uv run scripts/start_weather.py [--workers 8]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weather as wx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V4 = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"]


def _one(job):
    gy, gx, ct, pts = job  # pts: (index, row in chunk, col in chunk, hour in chunk)
    if wx.DS is None:
        wx._open()
    for attempt in range(4):
        try:
            sub = wx.DS[V4].isel(x=slice(gx * wx.CHUNK_XY, (gx + 1) * wx.CHUNK_XY), y=slice(gy * wx.CHUNK_XY, (gy + 1) * wx.CHUNK_XY),
                                 time=slice(ct * wx.CHUNK_T, (ct + 1) * wx.CHUNK_T)).load()
            break
        except Exception as err:  # noqa: BLE001
            if attempt == 3:
                print(f"FAILED {gy}_{gx}_{ct}: {err}", flush=True)
                return []
            time.sleep(3 + 5 * attempt)
    t, rh = sub.temperature_2m.values, sub.relative_humidity_2m.values
    ms = np.hypot(sub.wind_u_10m.values, sub.wind_v_10m.values)
    out = []
    for i, r, c, h in pts:
        if r >= t.shape[1] or c >= t.shape[2] or h >= t.shape[0]:
            continue
        s = slice(h, min(h + 73, t.shape[0]))
        tt, hh, ww = t[s, r, c], rh[s, r, c], ms[s, r, c]
        ff = wx.fosberg(wx.emc(hh, tt), ww * 2.2369363)
        if not np.isfinite(ff).any():
            continue
        out.append(dict(i=i, ff_0=float(ff[0]), ff_max72=float(np.nanmax(ff)), ff_mean72=float(np.nanmean(ff)),
                        rh_min72=float(np.nanmin(hh)), wind_max72=float(np.nanmax(ww)), t_0=float(tt[0]), hours72=int(len(ff))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import start_fuel
    from pyproj import Transformer
    f = start_fuel.fires(400)
    wx._open()
    times = wx.DS.time.values.astype("datetime64[h]")
    hour = (pd.to_datetime(f.discovered, utc=True).dt.tz_localize(None).dt.floor("h").to_numpy().astype("datetime64[h]") - times[0]).astype(int)
    x, y = Transformer.from_crs("EPSG:4326", wx.PROJ, always_xy=True).transform(f.lon.to_numpy(), f.lat.to_numpy())
    col, row = np.floor((x - wx.GRID["x0"]) / wx.GRID["d"]).astype(int), np.floor((wx.GRID["y0"] - y) / wx.GRID["d"]).astype(int)
    ok = (hour >= 0) & (hour < len(times)) & (col >= 0) & (col < wx.GRID["w"]) & (row >= 0) & (row < wx.GRID["h"])
    groups = {}
    for i in np.nonzero(ok)[0]:
        key = (int(row[i] // wx.CHUNK_XY), int(col[i] // wx.CHUNK_XY), int(hour[i] // wx.CHUNK_T))
        groups.setdefault(key, []).append((int(i), int(row[i] % wx.CHUNK_XY), int(col[i] % wx.CHUNK_XY), int(hour[i] % wx.CHUNK_T)))
    jobs = [(*k, v) for k, v in sorted(groups.items())]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{int(ok.sum())} of {len(f)} fires inside the weather record, {len(jobs)} chunks to read", flush=True)
    proc, t0, rows = psutil.Process(os.getpid()), time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, got in enumerate(ex.map(_one, jobs)):
            rows += got
            gb = (proc.memory_info().rss + sum(ch.memory_info().rss for ch in proc.children(recursive=True))) / 1e9
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}", flush=True)
                os._exit(2)
            if (k + 1) % 25 == 0 or k + 1 == len(jobs):
                el = time.time() - t0
                print(f"{k + 1}/{len(jobs)} chunks  {el / 60:.1f} min  {gb:.1f} GB  about {el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} min left", flush=True)
    d = pd.DataFrame(rows)
    d["irwin"] = f.irwin.to_numpy()[d.i.to_numpy()]
    d = d.drop(columns="i").round(2)
    d.to_parquet(ROOT / "results" / "start_weather.parquet", compression="zstd", index=False)
    print(d.describe().round(1).to_string())


if __name__ == "__main__":
    main()
