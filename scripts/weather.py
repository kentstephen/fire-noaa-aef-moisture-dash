"""Slow-moisture Fosberg over California from the HRRR analysis (dynamical.org Icechunk, 3 km, hourly).

Fosberg's index reads the moisture dead fuel would have if it matched the air at once. Here the moisture follows the
air with a time lag instead (the 10, 100 and 1000 hour fuel classes of the US fire danger system):

    m(t) = target(t) + (m(t-1) - target(t)) * exp(-1 / tau)

target is the air's equilibrium moisture (from 2 m temperature and humidity), or 30 % (the top of Fosberg's scale,
where the index reaches zero) in an hour with rain of RAIN_MM or more. Wind is the mean of the last 24 hours, so the
afternoon peak does not pulse. The index is Fosberg's own formula on that moisture and that wind.

Read one store chunk group at a time (45 x 45 pixels, 2160 hours, 5 variables: about 90 MB of float32 per time
chunk). Nothing is written to disk; daily() returns the fields in memory.
"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CHUNK_T, CHUNK_XY = 2160, 45
VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m", "precipitation_surface"]
TAUS = (10, 100, 1000)
RAIN_MM = 0.25
GRID = dict(x0=-2699020.142521929, y0=1588193.847443335, d=3000.0, w=1799, h=1059)
PROJ = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
DS = None


def emc(rh, t_c):
    t_f = t_c * 1.8 + 32
    h = np.clip(rh, 0, 100)
    m = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t_f,
                 np.where(h < 50, 2.22749 + 0.160107 * h - 0.014784 * t_f,
                          21.0606 + 0.005565 * h * h - 0.00035 * h * t_f - 0.483199 * h))
    return np.clip(m, 0, 30)


def fosberg(m, wind_mph):
    mr = m / 30
    return (1 - 2 * mr + 1.5 * mr**2 - 0.5 * mr**3) * np.sqrt(1 + wind_mph**2) / 0.3002


def _open():
    global DS
    import icechunk
    import xarray as xr
    storage = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk",
                                  region="us-west-2", anonymous=True)
    DS = xr.open_dataset(icechunk.Repository.open(storage).readonly_session("main").store, engine="zarr",
                         consolidated=False)[VARS]


def _group(job):
    """One 45 x 45 group over the time chunks ct0..ct1: the index at 00 UTC of each day, per tau."""
    gy, gx, ct0, ct1 = job
    if DS is None:
        _open()
    m = wind_hist = None
    out, days = [], []
    for ct in range(ct0, ct1 + 1):
        for attempt in range(4):
            try:
                sub = DS.isel(x=slice(gx * CHUNK_XY, (gx + 1) * CHUNK_XY), y=slice(gy * CHUNK_XY, (gy + 1) * CHUNK_XY),
                              time=slice(ct * CHUNK_T, (ct + 1) * CHUNK_T)).load()
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(3 + 5 * attempt)
        times = sub.time.values
        t, rh = sub.temperature_2m.values, sub.relative_humidity_2m.values
        wind = np.hypot(sub.wind_u_10m.values, sub.wind_v_10m.values) * 2.2369363
        rain = sub.precipitation_surface.values * 3600
        target = np.where(rain >= RAIN_MM, 30.0, emc(rh, t)).astype(np.float32)
        del sub, t, rh, rain
        n = len(times)
        target = target.reshape(n, -1)
        wind = wind.reshape(n, -1)
        if m is None:
            first = np.where(np.isfinite(target[0]), target[0], 10.0)
            m = np.stack([first] * len(TAUS))
            wind_hist = np.zeros((24, target.shape[1]), np.float32)
            wind_hist[:] = np.nan_to_num(wind[0])
        decay = np.exp(-1 / np.array(TAUS, np.float32))[:, None]
        for i in range(n):
            tg = target[i]
            ok = np.isfinite(tg)
            m = np.where(ok, tg + (m - tg) * decay, m)
            w = wind[i]
            wind_hist[i % 24] = np.where(np.isfinite(w), w, wind_hist[i % 24])
            if times[i].astype("datetime64[h]").astype(int) % 24 == 0:
                out.append(fosberg(m, wind_hist.mean(0)).astype(np.float32))
                days.append(times[i].astype("datetime64[D]"))
    return gy, gx, np.array(days), np.stack(out)  # (days, taus, 2025)


def window(lon0, lat0, lon1, lat1):
    """Chunk groups (gy0, gy1, gx0, gx1, inclusive) covering a lon/lat box."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
    lon = np.r_[np.linspace(lon0, lon1, 50), np.linspace(lon0, lon1, 50), [lon0] * 50, [lon1] * 50]
    lat = np.r_[[lat0] * 50, [lat1] * 50, np.linspace(lat0, lat1, 50), np.linspace(lat0, lat1, 50)]
    x, y = tr.transform(lon, lat)
    col, row = (x - GRID["x0"]) / GRID["d"], (GRID["y0"] - y) / GRID["d"]
    return (max(0, int(row.min()) // CHUNK_XY), min(GRID["h"] - 1, int(row.max())) // CHUNK_XY,
            max(0, int(col.min()) // CHUNK_XY), min(GRID["w"] - 1, int(col.max())) // CHUNK_XY)


def daily(box, time_chunks=3, workers=6, pts=None):
    """(days, field) with field (days, taus, rows, cols) float32 on the HRRR grid, plus the window's first row and
    column. The last `time_chunks` chunks of 2160 hours; the first one is spin-up for the slowest lag."""
    _open()
    last = (DS.sizes["time"] - 1) // CHUNK_T
    gy0, gy1, gx0, gx1 = window(*box)
    jobs = [(gy, gx, last - time_chunks + 1, last) for gy in range(gy0, gy1 + 1) for gx in range(gx0, gx1 + 1)]
    if pts is not None:  # only the groups that hold one of the points (lon, lat arrays), plus their neighbors' edge
        from pyproj import Transformer
        x, y = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True).transform(*pts)
        c, r = (x - GRID["x0"]) / GRID["d"], (GRID["y0"] - y) / GRID["d"]
        want = set()
        for dc in (-1, 1):
            for dr in (-1, 1):
                want |= set(zip(((r + dr) // CHUNK_XY).astype(int).tolist(), ((c + dc) // CHUNK_XY).astype(int).tolist()))
        jobs = [j for j in jobs if (j[0], j[1]) in want]
    field = days = None
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, (gy, gx, d, f) in enumerate(ex.map(_group, jobs)):
            if field is None:
                days = d
                field = np.full((len(d), len(TAUS), (gy1 - gy0 + 1) * CHUNK_XY, (gx1 - gx0 + 1) * CHUNK_XY), np.nan, np.float32)
            h = min(CHUNK_XY, GRID["h"] - gy * CHUNK_XY)
            w = min(CHUNK_XY, GRID["w"] - gx * CHUNK_XY)
            r, c = (gy - gy0) * CHUNK_XY, (gx - gx0) * CHUNK_XY
            field[:, :, r:r + h, c:c + w] = f.reshape(len(d), len(TAUS), h, w)
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                print(f"weather {k + 1}/{len(jobs)} groups  {time.time() - t0:.0f} s", flush=True)
    return days, field, gy0 * CHUNK_XY, gx0 * CHUNK_XY


def sample(field2d, row0, col0, lon, lat):
    """Bilinear read of one (rows, cols) field at lon/lat arrays."""
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True).transform(lon, lat)
    c = (x - GRID["x0"]) / GRID["d"] - 0.5 - col0
    r = (GRID["y0"] - y) / GRID["d"] - 0.5 - row0
    c = np.clip(c, 0, field2d.shape[1] - 1.001)
    r = np.clip(r, 0, field2d.shape[0] - 1.001)
    c0, r0 = c.astype(int), r.astype(int)
    fc, fr = c - c0, r - r0
    return (field2d[r0, c0] * (1 - fr) * (1 - fc) + field2d[r0, c0 + 1] * (1 - fr) * fc
            + field2d[r0 + 1, c0] * fr * (1 - fc) + field2d[r0 + 1, c0 + 1] * fr * fc)
