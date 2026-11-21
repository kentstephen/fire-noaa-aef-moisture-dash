# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pandas", "pyarrow", "pyproj", "psutil", "duckdb>=2.0.1",
#                 "scikit-learn"]
# ///
"""Do GOES fire detections fall where the map's risk is high? California, 2026.

Cases: the GOES-West fire events of aef-hls-s1-fire-knn (data/goes/events_<year>.parquet: fire pixels chained into
events, each labeled perimeter / start when a recorded wildfire is near, else unmatched). A case is an event's
first place and first hour.

What is read at each case and control, the way the web map computes it (web/src/risk.ts, wx.fc.worker.ts):
  ground     the fuel of the A5 resolution 11 cell (about 2.9 km, a GOES pixel) as the ground's logit
  weather    the hour's weather score (Fosberg, temperature, humidity, wind; logits)
  fosberg    the plain Fosberg index of the hour (wind and the air's dryness, no temperature term)
  risk       ground + weather where the weather is over the gate (what the ramp showed with the fitted score)
Dryness that builds up, from the same weather record (as scripts/weather.py: dead fuel moisture follows the air's
equilibrium moisture with a time lag, and an hour of rain pushes it to 30 %), spun up over the 90 days before:
  dry100, dry1000   minus the 100 hour and 1000 hour fuel moisture (higher = drier)
  fosberg_m100      Fosberg's formula on the 100 hour moisture and the hour's wind
  dry_days          days since an hour with 1 mm of rain or more
  calendar          days since the first hour read: what a plain date knows, to hold the dryness scores against
The weather is the HRRR analysis (dynamical.org), not the forecast the map draws: one continuous record for the
season, the same variables and the same formula.

Controls, each isolating one thing:
  same place, other day   same point and hour of day, another day 3 or more days away: only the weather differs
  same place, 3 weeks     the same, but 3 to 21 days away, so the season is nearly the same
  ring 2.5 / 5-10 / 20-30 km   same hour, a point that far away: only the ground differs (and less so when near)
  anywhere, same hour     a random California cell at the case's hour: what the state looked like then
  anywhere, other day     a random cell on the other day: both differ
Ground must read 0.5 against "same place" and weather about 0.5 against the near ring; anything else is a leak.

AUC: the chance a case scores higher than one of its controls. Intervals: 95 %, resampling 0.1 degree places.
Nothing is fit here; the scores are the map's own.

Peak RAM about 15 GB (six fields of hours x pixels as float16, 6 readers; 11.4 GB was measured with four). Stops past --max-gb.
Writes results/goes_overlap_<year>.json (a few KB).

    uv run scripts/goes_overlap.py [--year 2026] [--start 2026-05-01] [--end 2026-10-27]
"""
import argparse
import json
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
GOES = ROOT.parent / "aef-hls-s1-fire-knn" / "data" / "goes"
RES = 11
CA = (-124.6, 32.4, -113.9, 42.1)
V4 = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"]
# the weather part of the ignition head of aef-landcover-fosberg-cog: weight, mean, spread (as in wx.fc.worker.ts)
W = dict(ffwi=(-0.16230474855586324, 16.678741198055963, 13.197587332656665),
         t=(0.6628038301659839, 17.88455161384719, 10.032864575742538),
         rh=(-0.5549916574015389, 44.238219400525516, 24.157877178327944),
         wind=(0.36938999709829046, 3.848127975677117, 2.4595160400134612))
KEEP = float(np.exp(-1 / 48))
GROUND_LO, GROUND_HI, GATE, P_HIDE = -2.43, 2.8, 1.0, 0.5
VOL_LO, VOL_HI = np.log(0.02), np.log(50)
RINGS = {"ring 2.5 km": (2.5, 2.5), "ring 5-10 km": (5, 10), "ring 20-30 km": (20, 30)}
K_TIME, K_RING, K_ANY, K_NEAR = 5, 3, 5, 5
SCORES = ["ground", "weather", "fosberg", "dry100", "dry1000", "fosberg_m100", "dry_days", "calendar", "risk"]
FIELDS = ("weather", "fosberg", "m100", "m1000", "fosberg_m100", "dry_days")
V5 = V4 + ["precipitation_surface"]
FF_GATES = (20, 30, 40)


def z_of(k, v):
    return W[k][0] * (v - W[k][1]) / W[k][2]


def _hourly(job):
    """One 45 x 45 group over the time chunks ct0..ct1 (the chunk before is read too, to spin the moisture up): FIELDS."""
    gy, gx, ct0, ct1 = job
    if wx.DS is None:
        wx._open()
    outs, m, dry = [[] for _ in FIELDS], None, None
    decay = np.exp(-1 / np.array([100, 1000], np.float32))[:, None]
    for ct in range(max(0, ct0 - 1), ct1 + 1):
        for attempt in range(4):
            try:
                sub = wx.DS[V5].isel(x=slice(gx * wx.CHUNK_XY, (gx + 1) * wx.CHUNK_XY),
                                     y=slice(gy * wx.CHUNK_XY, (gy + 1) * wx.CHUNK_XY),
                                     time=slice(ct * wx.CHUNK_T, (ct + 1) * wx.CHUNK_T)).load()
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(3 + 5 * attempt)
        n = sub.sizes["time"]
        flat = lambda v: np.asarray(v, np.float32).reshape(n, -1)  # noqa: E731
        t, rh = flat(sub.temperature_2m.values), flat(sub.relative_humidity_2m.values)
        ms = flat(np.hypot(sub.wind_u_10m.values, sub.wind_v_10m.values))
        rain = flat(sub.precipitation_surface.values) * 3600
        del sub
        e = wx.emc(rh, t).astype(np.float32)
        ff = wx.fosberg(e, ms * 2.2369363).astype(np.float32)
        z = (z_of("ffwi", ff) + z_of("t", t) + z_of("rh", rh) + z_of("wind", ms)).astype(np.float32)
        target = np.where(rain >= wx.RAIN_MM, 30.0, e).astype(np.float32)
        if m is None:
            first = np.where(np.isfinite(target[0]), target[0], 10.0).astype(np.float32)
            m, dry = np.stack([first, first]), np.zeros(target.shape[1], np.float32)
        M, D = np.empty((2, n, target.shape[1]), np.float32), np.empty_like(target)
        for i in range(n):
            tg = target[i]
            m = np.where(np.isfinite(tg), tg + (m - tg) * decay, m)
            dry = np.where(rain[i] >= 1.0, 0.0, dry + 1 / 24)
            M[:, i], D[i] = m, dry
        if ct >= ct0:
            for o, v in zip(outs, (z, ff, M[0], M[1], wx.fosberg(M[0], ms * 2.2369363), D)):
                o.append(v.astype(np.float16))
    return gy, gx, [np.concatenate(o) for o in outs]


def fields(ct0, ct1, workers, max_gb):
    gy0, gy1, gx0, gx1 = wx.window(*CA)
    jobs = [(gy, gx, ct0, ct1) for gy in range(gy0, gy1 + 1) for gx in range(gx0, gx1 + 1)]
    F = None
    proc, t0 = psutil.Process(os.getpid()), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, (gy, gx, got) in enumerate(ex.map(_hourly, jobs)):
            if F is None:
                shape = (len(got[0]), (gy1 - gy0 + 1) * wx.CHUNK_XY, (gx1 - gx0 + 1) * wx.CHUNK_XY)
                F = [np.full(shape, np.nan, np.float16) for _ in got]
            h = min(wx.CHUNK_XY, wx.GRID["h"] - gy * wx.CHUNK_XY)
            w = min(wx.CHUNK_XY, wx.GRID["w"] - gx * wx.CHUNK_XY)
            r, c = (gy - gy0) * wx.CHUNK_XY, (gx - gx0) * wx.CHUNK_XY
            for fld, v in zip(F, got):
                fld[:, r:r + h, c:c + w] = v.reshape(len(v), h, w)
            gb = (proc.memory_info().rss + sum(ch.memory_info().rss for ch in proc.children(recursive=True))) / 1e9
            if gb > max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {max_gb}", flush=True)
                os._exit(2)
            if (k + 1) % 6 == 0 or k + 1 == len(jobs):
                print(f"weather {k + 1}/{len(jobs)} groups  {time.time() - t0:.0f} s  {gb:.1f} GB", flush=True)
    return F, gy0 * wx.CHUNK_XY, gx0 * wx.CHUNK_XY


def at(F, row0, col0, hour, lon, lat):
    """The field at each point's hour, read between the four pixels around it."""
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", wx.PROJ, always_xy=True).transform(lon, lat)
    c = np.clip((x - wx.GRID["x0"]) / wx.GRID["d"] - 0.5 - col0, 0, F.shape[2] - 1.001)
    r = np.clip((wx.GRID["y0"] - y) / wx.GRID["d"] - 0.5 - row0, 0, F.shape[1] - 1.001)
    c0, r0 = c.astype(int), r.astype(int)
    fc, fr = c - c0, r - r0
    g = lambda dr, dc: F[hour, r0 + dr, c0 + dc].astype(np.float32)  # noqa: E731
    return g(0, 0) * (1 - fr) * (1 - fc) + g(0, 1) * (1 - fr) * fc + g(1, 0) * fr * (1 - fc) + g(1, 1) * fr * fc


def auc(pos, neg):
    """Chance a case outscores a control, ties counting half."""
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-10-27")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    import duckdb
    con = duckdb.connect()
    con.execute("SET allow_extension_repositories = 'allowed'")
    con.execute("LOAD a5 FROM query_farm")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    fuel_file = ROOT / "web" / "public" / "data" / f"fuel_r{RES}.parquet"
    cells = con.execute(f"""SELECT cell, vol, ST_X(a5_cell_to_point(cell)) AS lon, ST_Y(a5_cell_to_point(cell)) AS lat
                            FROM '{fuel_file}' WHERE n > 0""").df()

    def fuel_at(lon, lat):
        q = pd.DataFrame(dict(i=np.arange(len(lon)), lon=lon, lat=lat))  # noqa: F841 (read by duckdb)
        got = con.execute(f"""SELECT q.i, f.vol FROM q LEFT JOIN '{fuel_file}' f
                              ON f.cell = a5_lonlat_to_cell(q.lon, q.lat, {RES}) AND f.n > 0 ORDER BY q.i""").df()
        return got.vol.to_numpy(np.float64)

    # the weather record: which hours, which store chunks
    wx._open()
    times = wx.DS.time.values.astype("datetime64[h]")
    i0 = int(np.searchsorted(times, np.datetime64(a.start, "h")))
    i1 = int(np.searchsorted(times, np.datetime64(a.end, "h") + np.timedelta64(24, "h"))) - 1
    ct0, ct1 = i0 // wx.CHUNK_T, min(i1, len(times) - 1) // wx.CHUNK_T
    base = ct0 * wx.CHUNK_T
    nh = min(len(times), (ct1 + 1) * wx.CHUNK_T) - base
    print(f"weather hours {times[base]} to {times[base + nh - 1]} UTC ({ct1 - ct0 + 1} store chunks, {nh} hours)", flush=True)

    ev = pd.read_parquet(GOES / f"events_{a.year}.parquet")
    ev["hour"] = ((ev.t0.dt.tz_convert("UTC").dt.tz_localize(None).dt.round("h").to_numpy().astype("datetime64[h]")
                   - times[base]).astype(int))
    lo, hi = max(i0 - base, 72), min(i1 - base, nh - 1)  # 72 h for the sustained score to settle
    ev = ev[(ev.hour >= lo) & (ev.hour <= hi)].reset_index(drop=True)
    ev["vol"] = fuel_at(ev.lon.to_numpy(), ev.lat.to_numpy())
    n_all = len(ev)
    ev = ev[ev.vol.notna()].reset_index(drop=True)
    print(f"{n_all} events in the hours, {len(ev)} on a baked California cell "
          f"({int((ev.label != 'unmatched').sum())} with a recorded wildfire)", flush=True)

    # cases and controls as one table of points
    n = len(ev)
    rows = [pd.DataFrame(dict(case=ev.event, kind="case", lon=ev.lon, lat=ev.lat, hour=ev.hour, vol=ev.vol))]
    # same place, other day: same hour of day, 3 or more days away
    days = (hi - lo) // 24
    off = rng.integers(3, max(4, days), (n, K_TIME * 6)) * rng.choice([-1, 1], (n, K_TIME * 6))
    other = np.full((n, K_TIME), -1)
    for i in range(n):
        h = ev.hour[i] + 24 * off[i]
        h = np.unique(h[(h >= lo) & (h <= hi)])
        h = rng.permutation(h)[:K_TIME]
        other[i, :len(h)] = h
    for k in range(K_TIME):
        ok = other[:, k] >= 0
        rows.append(pd.DataFrame(dict(case=ev.event[ok], kind="same place, other day", lon=ev.lon[ok], lat=ev.lat[ok],
                                      hour=other[ok, k], vol=ev.vol[ok])))
    near = rng.integers(3, 22, (n, K_NEAR)) * rng.choice([-1, 1], (n, K_NEAR))
    for k in range(K_NEAR):
        h = ev.hour.to_numpy() + 24 * near[:, k]
        ok = (h >= lo) & (h <= hi)
        rows.append(pd.DataFrame(dict(case=ev.event[ok], kind="same place, 3 weeks", lon=ev.lon[ok], lat=ev.lat[ok],
                                      hour=h[ok], vol=ev.vol[ok])))
    # rings: same hour, a point at a distance, on a baked California cell
    tries = 12
    for name, (d0, d1) in RINGS.items():
        d = rng.uniform(d0, d1, (n, tries))
        b = rng.uniform(0, 2 * np.pi, (n, tries))
        lat = ev.lat.to_numpy()[:, None] + d * np.cos(b) / 111.2
        lon = ev.lon.to_numpy()[:, None] + d * np.sin(b) / (111.2 * np.cos(np.radians(ev.lat.to_numpy()))[:, None])
        v = fuel_at(lon.ravel(), lat.ravel()).reshape(n, tries)
        rank = np.cumsum(np.isfinite(v), axis=1)
        for k in range(1, K_RING + 1):
            j = np.argmax(rank == k, axis=1)
            ok = rank[np.arange(n), j] == k
            ii = np.nonzero(ok)[0]
            rows.append(pd.DataFrame(dict(case=ev.event[ii], kind=name, lon=lon[ii, j[ii]], lat=lat[ii, j[ii]],
                                          hour=ev.hour[ii], vol=v[ii, j[ii]])))
    # anywhere in California: at the case's hour, and on the other day
    for k in range(K_ANY):
        pick = cells.iloc[rng.integers(0, len(cells), n)].reset_index(drop=True)
        rows.append(pd.DataFrame(dict(case=ev.event, kind="anywhere, same hour", lon=pick.lon, lat=pick.lat, hour=ev.hour, vol=pick.vol)))
        ok = other[:, k] >= 0
        pick = cells.iloc[rng.integers(0, len(cells), n)].reset_index(drop=True)
        rows.append(pd.DataFrame(dict(case=ev.event[ok], kind="anywhere, other day", lon=pick.lon[ok], lat=pick.lat[ok],
                                      hour=other[ok, k], vol=pick.vol[ok])))
    pts = pd.concat(rows, ignore_index=True)

    F, row0, col0 = fields(ct0, ct1, a.workers, a.max_gb)
    hour, lon, lat = pts.hour.to_numpy(int), pts.lon.to_numpy(), pts.lat.to_numpy()
    for name, fld in zip(FIELDS, F):
        pts[name] = at(fld, row0, col0, hour, lon, lat)
    del F
    pts["dry100"], pts["dry1000"], pts["calendar"] = -pts.m100, -pts.m1000, pts.hour / 24.0
    fuel = np.clip((np.log(0.02 + pts.vol.to_numpy()) - VOL_LO) / (VOL_HI - VOL_LO), 0, 1)
    pts["ground"] = GROUND_LO + (GROUND_HI - GROUND_LO) * fuel
    u = np.clip((pts.weather - GATE) / 0.5, 0, 1)
    pts["lit"] = u * u * (3 - 2 * u)
    pts["risk"] = pts.ground + pts.weather * pts.lit
    pts = pts.dropna(subset=list(FIELDS)).reset_index(drop=True)
    meta = ev.set_index("event")
    pts["wild"] = (meta.label.reindex(pts.case).to_numpy() != "unmatched")
    pts["place"] = (np.floor(meta.lon.reindex(pts.case).to_numpy() * 10).astype(int) * 10000
                    + np.floor(meta.lat.reindex(pts.case).to_numpy() * 10).astype(int))

    out = dict(year=a.year, hours=[str(times[base + lo]), str(times[base + hi])], events=int(len(ev)),
               wildfires=int((ev.label != "unmatched").sum()), res=RES, gate=GATE, auc={}, share={})
    kinds = [k for k in pts.kind.unique() if k != "case"]
    for group, sel in (("recorded wildfires", pts.wild), ("all GOES events", np.ones(len(pts), bool))):
        g = pts[sel]
        cases = g[g.kind == "case"]
        # how much of it the map had lit: the cases next to the state at the same hours
        sh = {}
        med = lambda d, k: round(float(d[k].median()), 1)  # noqa: E731
        for name, d in (("at the fires", cases), ("same place, other day", g[g.kind == "same place, other day"]),
                        ("same place, 3 weeks", g[g.kind == "same place, 3 weeks"]), ("California, same hours", g[g.kind == "anywhere, same hour"])):
            sh[name] = dict(n=int(len(d)), fuel_m=round(float(d.vol.median()), 2), fosberg=med(d, "fosberg"),
                            **{f"fosberg_over_{g_}": round(float((d.fosberg >= g_).mean()), 3) for g_ in FF_GATES},
                            m100=med(d, "m100"), m1000=med(d, "m1000"), fosberg_m100=med(d, "fosberg_m100"), dry_days=med(d, "dry_days"),
                            m100_under_8=round(float((d.m100 < 8).mean()), 3), m1000_under_10=round(float((d.m1000 < 10).mean()), 3))
        out["share"][group] = sh
        res = {}
        places = cases.place.unique()
        for kind in kinds:
            ctl = g[g.kind == kind]
            by_c = {p: d for p, d in cases.groupby("place")}
            by_k = {p: d for p, d in ctl.groupby("place")}
            res[kind] = {}
            for s in SCORES:
                boots = []
                pc = {p: d[s].to_numpy() for p, d in by_c.items()}
                pk = {p: d[s].to_numpy() for p, d in by_k.items()}
                for _ in range(a.boot):
                    draw = rng.choice(places, len(places))
                    pos = np.concatenate([pc[p] for p in draw])
                    neg = np.concatenate([pk[p] for p in draw if p in pk])
                    boots.append(auc(pos, neg))
                res[kind][s] = [round(auc(cases[s].to_numpy(), ctl[s].to_numpy()), 3),
                                round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]
        out["auc"][group] = res
        print(f"\n{group}: {len(cases)} cases in {len(places)} places")
        for name, v in sh.items():
            print(f"  {name:24s} {v}")
        print(f"  {'control':24s}" + "".join(f"{s:>13s}" for s in SCORES))
        for kind in kinds:
            print(f"  {kind:24s}" + "".join(f"{v[0]:>13.3f}" for v in (res[kind][s] for s in SCORES)))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"goes_overlap_{a.year}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
