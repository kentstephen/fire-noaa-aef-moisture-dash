# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pandas", "pyproj", "psutil", "duckdb>=2.0.1", "shapely",
#                 "pillow", "pyarrow"]
# ///
"""Stills of California: A5 cells, brightness from fuel, color from slow-moisture Fosberg.

Every image pixel is given the A5 cell its center falls in, so cells are drawn in their own shape at any scale.
  brightness  the cell's mean vegetation volume (the volume head at 10 m, averaged; data/parts from bake_fuel.py)
  saturation  Fosberg on time-lagged fuel moisture and 24 hour mean wind (weather.py), read at the pixel
Gray is fuel that the weather has not primed; orange is fuel that it has. Ground with no fuel stays dark.

Writes PNGs to --out. The weather is read into memory each run (nothing kept on disk). RAM: about 1 GB per
weather worker plus about 2 GB here; stops over --max-gb.

    uv run scripts/still.py --out DIR [--fake-weather] [--zoom LON LAT KM]
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weather as wx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "data" / "parts"
BOX = (-124.5, 32.45, -114.05, 42.1)
BASE_RES = 15
VOL_TOP = 16.0    # volume (m) drawn at full brightness
F_LO, F_HI = 8.0, 30.0   # index drawn gray, index drawn at full color
BG = (11, 13, 16)


def guard(max_gb):
    import psutil

    def run():
        p = psutil.Process()
        while True:
            if p.memory_info().rss / 1e9 > max_gb:
                print("STOP: memory", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


def merc(lat):
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def unmerc(y):
    return np.degrees(2 * np.arctan(np.exp(np.radians(y))) - np.pi / 2)


def grid(box, width):
    """lon, lat of every pixel center of a Web Mercator image of the box, and its height."""
    lon0, lat0, lon1, lat1 = box
    y0, y1 = merc(lat0), merc(lat1)
    height = int(round(width * (y1 - y0) / (lon1 - lon0)))
    lon = lon0 + (np.arange(width) + 0.5) / width * (lon1 - lon0)
    lat = unmerc(y1 - (np.arange(height) + 0.5) / height * (y1 - y0))
    return np.broadcast_to(lon, (height, width)).ravel(), np.broadcast_to(lat[:, None], (height, width)).ravel(), height


def res_for(box, width):
    """The A5 resolution whose cells are about 1.5 pixels wide, no finer than the bake."""
    px_m = (box[2] - box[0]) / width * 111320 * np.cos(np.radians((box[1] + box[3]) / 2))
    area = (1.5 * px_m) ** 2
    res = int(round(1 + np.log(5.1e14 / (60 * area)) / np.log(4)))
    return max(5, min(BASE_RES, res)), px_m


def fuel(con, lon, lat, res):
    """Mean volume of the res cell under each pixel (NaN where the bake has nothing)."""
    import pyarrow as pa
    con.register("px", pa.table({"i": np.arange(len(lon), dtype=np.int32), "lon": lon, "lat": lat}))
    parent = "cell" if res == BASE_RES else f"a5_cell_to_parent(cell, {res})"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE f AS SELECT {parent} AS c, sum(vol) / sum(n) AS v
                    FROM '{PARTS}/*.parquet' GROUP BY 1""")
    r = con.execute(f"""SELECT i, v FROM (SELECT i, a5_lonlat_to_cell(lon, lat, {res}) AS c FROM px) p
                        LEFT JOIN f USING (c)""").fetchnumpy()
    out = np.full(len(lon), np.nan, np.float32)
    out[r["i"]] = np.ma.filled(r["v"].astype(np.float32), np.nan) if np.ma.isMaskedArray(r["v"]) else r["v"]
    con.unregister("px")
    return out


def inside(lon, lat):
    import shapely
    ca = shapely.union_all([shapely.Polygon(r) for r in json.load(open(ROOT / "data" / "ca.json"))])
    shapely.prepare(ca)
    return shapely.contains_xy(ca, lon, lat)


def oklch(L, C, hue):
    a, b = C * np.cos(np.radians(hue)), C * np.sin(np.radians(hue))
    l_, m_, s_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, \
        (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.stack([4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
                    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
                    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_], -1).clip(0, 1)
    return (np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055) * 255).round().astype(np.uint8)


def ramp50(v):
    """The long orange ramp (L, C, hue) for v in 0..1, in the manner of the 50 stop ramps: orange's own path with
    darker steps below it toward near-black and paler steps above it toward a pale lemon."""
    L = 0.11 + 0.83 * v
    hue = 58 + 44 * v
    # chroma held under what the lightness can carry, so the dark steps are browns and never clip to red
    C = np.minimum(0.19 * np.sin(np.pi * np.clip(v, 0, 1) ** 0.85) ** 0.6 * (1 - 0.35 * v), 0.36 * L)
    return L, C, hue


MODES = {
    "base": "gray by fuel, orange to yellow as it primes",
    "primed50": "primed side extended: deep orange to pale lemon",
    "one50": "one long ramp on fuel x primed",
    "both": "gray by fuel, long ramp takes over as it primes",
}


def paint(vol, f, keep, height, width, mode="base"):
    """Brightness from volume; the index takes the cell from gray into orange and on toward yellow.

    A cell only gets brighter as it gets more primed: orange at equal lightness reads darker than gray (more so with
    weak red vision), so lightness rises with the index and the hue moves from orange to yellow, never toward red.
    White is left for fire. Unprimed fuel is a dim gray. See MODES for the variants.
    """
    t = np.clip(np.nan_to_num(vol) / VOL_TOP, 0, 1) ** 0.5
    s = np.clip((np.nan_to_num(f) - F_LO) / (F_HI - F_LO), 0, 1)
    gray = 0.17 + 0.46 * t
    if mode == "base":
        hue = 62 + 33 * s
        L = gray * (0.72 + 0.28 * s) + 0.26 * s ** 0.7 * t
        C = 0.18 * np.minimum(1, 2.5 * s) ** 0.8 * (0.05 + 0.95 * t)
    elif mode == "primed50":
        hue = 56 + 46 * s
        L = gray * (0.72 + 0.28 * s) + 0.30 * s * t + 0.05 * s
        C = 0.20 * np.minimum(1, 3 * s) ** 0.8 * (0.05 + 0.95 * t) * (1 - 0.55 * np.clip((s - 0.7) / 0.3, 0, 1))
    elif mode == "one50":
        L, C, hue = ramp50(t * (0.10 + 0.90 * s))
    else:
        Lr, Cr, hue = ramp50(np.clip(s * t ** 0.6, 0, 1))
        w = np.clip(s / 0.3, 0, 1)
        w = w * w * (3 - 2 * w)
        L = (1 - w) * gray * 0.72 + w * np.maximum(Lr, gray * 0.72)
        C = w * Cr * (0.25 + 0.75 * t)
    rgb = oklch(L, C, hue)
    rgb[~(keep & np.isfinite(vol))] = BG
    return rgb.reshape(height, width, 3)


def key(img, mode, xy):
    """A small swatch of the encoding: fuel up, primed to the right."""
    from PIL import Image
    w, h = 220, 130
    vol = np.broadcast_to((np.linspace(1, 0, h) ** 2 * VOL_TOP)[:, None], (h, w)).ravel()
    f = np.broadcast_to(np.linspace(F_LO, F_HI, w), (h, w)).ravel()
    img.paste(Image.fromarray(paint(vol, f, np.ones(h * w, bool), h, w, mode)), xy)
    label(img, "more fuel", (xy[0], xy[1] - 26), 17, (150, 154, 160))
    label(img, "more primed", (xy[0] + w - 92, xy[1] + h + 6), 17, (150, 154, 160))


def when(day):
    """00 UTC of a day as Pacific and Eastern clock time."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    u = datetime.fromisoformat(str(day)).replace(tzinfo=timezone.utc)
    p, e = u.astimezone(ZoneInfo("America/Los_Angeles")), u.astimezone(ZoneInfo("America/New_York"))
    return f"{p:%b} {p.day}, {p.year}, {p:%-I %p} {p:%Z} ({e:%-I %p} {e:%Z})"


def label(img, text, xy, size=22, fill=(200, 204, 210)):
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text(xy, text, fill=fill, font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--fake-weather", action="store_true")
    ap.add_argument("--zoom", nargs=3, type=float, action="append", default=[], metavar=("LON", "LAT", "KM"))
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--zoom-mode", default="base", choices=list(MODES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=6.0)
    a = ap.parse_args()
    guard(a.max_gb)
    import duckdb
    from PIL import Image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    # A5 for DuckDB 2.0.1 and later comes from its author's repository (installed once with INSTALL a5 FROM query_farm)
    con.execute("SET allow_extension_repositories = 'allowed'")
    con.execute("LOAD a5 FROM query_farm")

    lon, lat, height = grid(BOX, a.width)
    keep = inside(lon, lat)
    res, px_m = res_for(BOX, a.width)
    t0 = time.time()
    vol = fuel(con, lon, lat, res)
    print(f"state view: {a.width} x {height}, {px_m:.0f} m pixels, A5 resolution {res}, fuel in {time.time() - t0:.0f} s; "
          f"volume percentiles 50/90/99: {np.nanpercentile(vol[keep], [50, 90, 99]).round(2)}", flush=True)

    if a.fake_weather:
        days = np.array(["2026-01-01"], "datetime64[D]")
        field = np.tile(np.linspace(0, 40, 400, dtype=np.float32), (1, 3, 400, 1))
        row0, col0 = 250, 100
    else:
        days, field, row0, col0 = wx.daily(BOX, workers=a.workers, pts=(lon[keep][::5], lat[keep][::5]))
    # how much of the state's fuel is lit each day, per lag
    t = np.clip(np.nan_to_num(vol) / VOL_TOP, 0, 1) ** 0.5 * keep
    sub = np.arange(0, len(lon), 7)
    lit = np.zeros((len(days), len(wx.TAUS)))
    for d in range(len(days)):
        for k in range(len(wx.TAUS)):
            f = wx.sample(field[d, k], row0, col0, lon[sub], lat[sub])
            lit[d, k] = (t[sub] * np.clip((np.nan_to_num(f) - F_LO) / (F_HI - F_LO), 0, 1)).sum() / t[sub].sum()
    start = min(len(days) - 1, 92)  # after the spin-up chunk
    for d in range(start, len(days)):
        print(days[d], " ".join(f"{v:.3f}" for v in lit[d]), flush=True)
    picks = [int(np.argmax(lit[start:, 1])) + start] if not a.days else \
        [int(np.where(days == np.datetime64(x))[0][0]) for x in a.days]

    for d in picks:
        panels = []
        for k, tau in enumerate(wx.TAUS):
            f = wx.sample(field[d, k], row0, col0, lon, lat)
            img = Image.fromarray(paint(vol, f, keep, height, a.width))
            label(img, f"{tau} hour fuel moisture", (24, height - 96), 26)
            key(img, "base", (a.width - 280, 70))
            label(img, when(days[d]), (24, height - 60), 20, (150, 154, 160))
            panels.append(img)
        sheet = Image.new("RGB", (a.width * 3, height), BG)
        for k, p in enumerate(panels):
            sheet.paste(p, (k * a.width, 0))
        sheet.save(out / f"ca_{days[d]}_lags.png")
        f = wx.sample(field[d, 1], row0, col0, lon, lat)
        four = Image.new("RGB", (a.width * 2, height * 2), BG)
        for k, (mode, text) in enumerate(MODES.items()):
            img = Image.fromarray(paint(vol, f, keep, height, a.width, mode))
            label(img, text, (24, height - 96), 26)
            label(img, "100 hour, " + when(days[d]), (24, height - 60), 20, (150, 154, 160))
            key(img, mode, (a.width - 280, 70))
            img.save(out / f"ca_{days[d]}_{mode}.png")
            four.paste(img, (k % 2 * a.width, k // 2 * height))
        four.save(out / f"ca_{days[d]}_ramps.png")
        for zlon, zlat, km in a.zoom:
            dlat = km / 2 / 111.32
            dlon = dlat / np.cos(np.radians(zlat))
            zbox = (zlon - dlon, zlat - dlat, zlon + dlon, zlat + dlat)
            zl, zt, zh = grid(zbox, 1400)
            zres, zpx = res_for(zbox, 1400)
            zv = fuel(con, zl, zt, zres)
            f = wx.sample(field[d, 1], row0, col0, zl, zt)
            img = Image.fromarray(paint(zv, f, np.ones(len(zl), bool), zh, 1400, a.zoom_mode))
            label(img, f"{km:.0f} km across, A5 resolution {zres}, 100 hour, {when(days[d])}", (24, zh - 60), 22)
            name = out / f"zoom_{zlon}_{zlat}_{km:.0f}km_{days[d]}.png"
            img.save(name)
            print("wrote", name, f"({zpx:.0f} m pixels, resolution {zres})", flush=True)


if __name__ == "__main__":
    main()
