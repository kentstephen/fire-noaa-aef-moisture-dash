# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyarrow", "torch", "psutil"]
# ///
"""The volume head fit for all of CONUS: the same small network as the California head, on windows from the whole
lower 48.

Rows: the national windows of scripts/sample_conus.py (data/conus) and, when the folder is there, the California
windows the first head was fit on (aef-hls-s1-fire-knn data/change, its random and burn windows). Every LANDFIRE
edition (2016, 2022, 2024, 2025) is a row with the embedding of its own year (LF2016 with 2017).

The network and its targets are those of aef-hls-s1-fire-knn scripts/fit_volume.py, so the export is read by the
same code: the year's embedding (64) -> 64 -> 5 values (log volume, lifeform cover and height, canopy cover and
height) + 11 fuel kinds + 4 lifeforms. Volume is LANDFIRE's leading lifeform's cover x height in m.

Scores, 5 folds by 0.5 degree block so neighbors are never on both sides: R^2 of log volume, the median error of
volume as a ratio, lifeform accuracy; for all rows and by region. Beside them, the California head
(models/fuel_volume.json) on the same rows, which is what the map would show there today.

Writes results/fit_conus.json and models/fuel_volume_conus.json (the national head, in the California head's format).

RAM: 64 float32 per row; 25 M rows is 6.3 GB, about 10 GB at the peak. Stops past --max-gb.

    uv run scripts/fit_conus.py [--epochs 12] [--no-cv] [--ca-dir PATH]
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
EDS = [2016, 2022, 2024, 2025]
REG = ["vol", "vc", "vh", "cc", "ch"]
LIFE = ["none", "tree", "shrub", "herb"]
CA_HEAD = json.loads((ROOT / "models" / "fuel_volume.json").read_text())
KINDS = CA_HEAD["kinds"]
NONFUEL = {91: 0, 92: 1, 93: 2, 98: 3, 99: 4}
FAMILY = [(101, 109, 5), (121, 124, 6), (141, 149, 7), (161, 165, 8), (181, 189, 9), (201, 204, 10)]
VOL_LO, VOL_HI = np.log(0.02), np.log(50.0)
RES, X0, Y0, CHUNK = 8.983111749910169e-05, -180.0, 83.68570533713473, 256
REGIONS = ["California", "West", "Central", "East"]
torch.set_num_threads(8)


def vol_scale(v):
    return (np.log(0.02 + v) - VOL_LO) / (VOL_HI - VOL_LO)


def vol_unscale(s):
    return np.exp(s * (VOL_HI - VOL_LO) + VOL_LO) - 0.02


def deq(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def kind_of(fb):
    k = np.full(fb.shape, -1, np.int64)
    for c, i in NONFUEL.items():
        k[fb == c] = i
    for lo, hi, i in FAMILY:
        k[(fb >= lo) & (fb <= hi)] = i
    return k


def lifeform(evc, evh):
    """EVC / EVH codes -> lifeform index, cover %, height m (NaN where LANDFIRE has no value)."""
    evc, evh = evc.astype(np.int32), evh.astype(np.int32)
    life = np.full(evc.shape, -1, np.int64)
    vc = np.full(evc.shape, np.nan, np.float32)
    vh = np.full(evc.shape, np.nan, np.float32)
    non = (evc >= 11) & (evc < 100)
    life[non], vc[non], vh[non] = 0, 0, 0
    for i, lo in ((1, 100), (2, 200), (3, 300)):
        m = (evc > lo) & (evc < lo + 100)
        life[m] = i
        vc[m] = evc[m] - lo
    for lo, scale in ((100, 1.0), (200, 0.1), (300, 0.1)):
        m = (evh > lo) & (evh < lo + 100) & (life > 0)
        vh[m] = (evh[m] - lo) * scale
    return life, vc, vh


def targets(z, ed):
    life, vc, vh = lifeform(z[f"evc_{ed}"], z[f"evh_{ed}"])
    c, h = z[f"cc_{ed}"].astype(np.float32), z[f"ch_{ed}"].astype(np.float32)
    cc = np.where((c >= 0) & (c <= 100), c, np.nan)
    ch = np.where((h >= 0) & (h <= 600), h / 10, np.nan)
    y = np.c_[vol_scale(vc / 100 * vh), vc / 100, vh / 40, cc / 100, ch / 50].astype(np.float32)
    return y, kind_of(z[f"fb_{ed}"]), life


def windows(ca_dir):
    """(path, cy, cx, the place of each edition's year in the file's q) for every training window."""
    out = []
    w = pq.read_table(ROOT / "data" / "conus" / "windows.parquet").to_pydict()
    for cy, cx in zip(w["cy"], w["cx"]):
        out.append((ROOT / "data" / "conus" / f"w_{cy}_{cx}.npz", cy, cx, [0, 1, 2, 3]))
    if ca_dir.exists():
        # the California files hold all nine years, 2017 first
        w = pq.read_table(ca_dir / "windows.parquet").to_pydict()
        for kind, cy, cx in zip(w["kind"], w["cy"], w["cx"]):
            if kind in ("random", "burn"):
                out.append((ca_dir / f"{kind}_{cy}_{cx}.npz", cy, cx, [0, 5, 7, 8]))
    else:
        print(f"no California windows at {ca_dir}", flush=True)
    return out


def table(ca_dir):
    X, Y, K, L, lon, lat, isca = [], [], [], [], [], [], []
    n_win = 0
    for path, cy, cx, yi in windows(ca_dir):
        if not path.exists():
            continue
        z = np.load(path)
        if len(z["q"]) == 0 or "evc_2025" not in z.files:
            continue
        n_win += 1
        z = {k: z[k] for k in z.files}
        for ed, i in zip(EDS, yi):
            y, k, life = targets(z, ed)
            X.append(deq(z["q"][:, i]))
            Y.append(y)
            K.append(k)
            L.append(life)
            n = len(y)
            lon.append(np.full(n, X0 + (cx + 0.5) * CHUNK * RES, np.float32))
            lat.append(np.full(n, Y0 - (cy + 0.5) * CHUNK * RES, np.float32))
            isca.append(np.full(n, len(yi) and yi[1] == 5))
    print(f"{n_win} windows with pixels", flush=True)
    return tuple(np.concatenate(a) for a in (X, Y, K, L, lon, lat, isca))


class Net(torch.nn.Module):
    def __init__(self, d=64, h=64):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(d, h), torch.nn.Linear(h, len(REG) + len(KINDS) + len(LIFE))

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def fit(Xs, idx, Y, K, L, epochs, seed=0):
    rng = np.random.default_rng(seed)
    Yt = torch.from_numpy(np.nan_to_num(Y, nan=-9.0))
    Kt, Lt = torch.from_numpy(K), torch.from_numpy(L)
    torch.manual_seed(seed)
    net = Net(Xs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    nr, nk = len(REG), len(KINDS)
    for _ in range(epochs):
        perm = torch.from_numpy(idx[rng.permutation(len(idx))])
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            o = net(Xs[b])
            y = Yt[b]
            m = y > -8
            loss = (((o[:, :nr] - y) ** 2) * m).sum() / m.sum().clamp(min=1) * 10
            mk, ml = Kt[b] >= 0, Lt[b] >= 0
            loss = loss + torch.nn.functional.cross_entropy(o[mk, nr:nr + nk], Kt[b][mk])
            loss = loss + torch.nn.functional.cross_entropy(o[ml, nr + nk:], Lt[b][ml])
            opt.zero_grad()
            loss.backward()
            opt.step()
        for g in opt.param_groups:
            g["lr"] *= 0.8
    net.eval()
    return net


def predict(net, Xs, idx):
    """Scaled log volume and lifeform for the rows idx."""
    vol, life = np.empty(len(idx), np.float32), np.empty(len(idx), np.int64)
    nr, nk = len(REG), len(KINDS)
    with torch.no_grad():
        for i in range(0, len(idx), 262144):
            o = net(Xs[torch.from_numpy(idx[i:i + 262144])]).numpy()
            vol[i:i + 262144], life[i:i + 262144] = o[:, 0], o[:, nr + nk:].argmax(1)
    return vol, life


def ca_head(X, mu_now, sd_now):
    """The California head's scaled log volume and lifeform on the same rows (X is standardized by mu_now, sd_now)."""
    mu, sd = np.array(CA_HEAD["mean"], np.float32), np.array(CA_HEAD["scale"], np.float32)
    w1 = np.array(CA_HEAD["w1"], np.float32).reshape(CA_HEAD["hidden"], 64)
    w2 = np.array(CA_HEAD["w2"], np.float32).reshape(CA_HEAD["n_out"], CA_HEAD["hidden"])
    b1, b2 = np.array(CA_HEAD["b1"], np.float32), np.array(CA_HEAD["b2"], np.float32)
    vol, life = np.empty(len(X), np.float32), np.empty(len(X), np.int64)
    for i in range(0, len(X), 262144):
        x = (X[i:i + 262144].numpy() * sd_now + mu_now - mu) / sd
        o = np.maximum(x @ w1.T + b1, 0) @ w2.T + b2
        vol[i:i + 262144], life[i:i + 262144] = o[:, 0], o[:, 16:20].argmax(1)
    return vol, life


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def score(Y, L, pv, pl, sel):
    y = Y[sel, 0]
    m = np.isfinite(y)
    v, p = vol_unscale(y[m]), vol_unscale(pv[sel][m]).clip(0)
    ml = L[sel] >= 0
    return dict(rows=int(sel.sum()), vol_r2=round(r2(y[m], pv[sel][m]), 3),
                vol_ratio_err=round(float(np.exp(np.median(np.abs(np.log((0.02 + v) / (0.02 + p)))))), 2),
                life_acc=round(float((pl[sel][ml] == L[sel][ml]).mean()), 3),
                vol_mean_landfire=round(float(v.mean()), 2), vol_mean_head=round(float(p.mean()), 2))


def export(net, mu, sd, path):
    s = net.state_dict()
    w1, b1, w2, b2 = (s[k].numpy() for k in ("l1.weight", "l1.bias", "l2.weight", "l2.bias"))
    q = np.random.default_rng(0).integers(-127, 128, (64, 64)).astype(np.int8)
    o = np.maximum((deq(q) - mu) / sd @ w1.T + b1, 0) @ w2.T + b2
    r = lambda a: [round(float(v), 6) for v in np.ravel(a)]  # noqa: E731
    path.write_text(json.dumps(dict(name="fuel_volume_conus", n_in=64, hidden=int(w1.shape[0]), n_out=int(w2.shape[0]), reg=REG,
                                    kinds=KINDS, life=LIFE, vol_lo=float(VOL_LO), vol_hi=float(VOL_HI), mean=r(mu), scale=r(sd),
                                    w1=r(w1), b1=r(b1), w2=r(w2), b2=r(b2), probe_q=q.ravel().tolist(), probe_out=r(o))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--max-gb", type=float, default=24.0)
    ap.add_argument("--ca-dir", type=Path, default=ROOT.parent / "aef-hls-s1-fire-knn" / "data" / "change")
    a = ap.parse_args()

    def watch():
        p = psutil.Process(os.getpid())
        while True:
            if p.memory_info().rss / 1e9 > a.max_gb:
                print(f"STOP: rss past --max-gb {a.max_gb}", flush=True)
                os._exit(2)
            time.sleep(1)

    threading.Thread(target=watch, daemon=True).start()
    rss = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e9  # noqa: E731
    t0 = time.time()
    X, Y, K, L, lon, lat, isca = table(a.ca_dir)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X -= mu
    X /= sd
    Xs = torch.from_numpy(X)
    region = np.where(isca, 0, np.where(lon < -104, 1, np.where(lon < -90, 2, 3)))
    print(f"{len(X):,} rows  " + "  ".join(f"{REGIONS[i]} {int((region == i).sum()):,}" for i in range(4)) + f"  rss {rss():.1f} GB", flush=True)
    res = {"rows": int(len(X))}
    if not a.no_cv:
        block = (np.floor(lon / 0.5) * 1000 + np.floor(lat / 0.5)).astype(np.int64)
        ub = np.unique(block)
        fold = dict(zip(np.random.default_rng(0).permutation(ub).tolist(), (np.arange(len(ub)) % 5).tolist()))
        fold = np.array([fold[b] for b in ub])[np.searchsorted(ub, block)]
        pv, pl = np.empty(len(X), np.float32), np.empty(len(X), np.int64)
        for f in range(5):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            net = fit(Xs, tr, Y, K, L, a.epochs, seed=f)
            pv[te], pl[te] = predict(net, Xs, te)
            print(f"  fold {f} done  {(time.time() - t0) / 60:.1f} min  rss {rss():.1f} GB", flush=True)
        cv, cl = ca_head(Xs, mu, sd)
        res["national head, held out"] = {"all": score(Y, L, pv, pl, np.ones(len(X), bool))}
        res["California head"] = {"all": score(Y, L, cv, cl, np.ones(len(X), bool))}
        for i, name in enumerate(REGIONS):
            if (region == i).any():
                res["national head, held out"][name] = score(Y, L, pv, pl, region == i)
                res["California head"][name] = score(Y, L, cv, cl, region == i)
        print(json.dumps(res, indent=1), flush=True)
    net = fit(Xs, np.arange(len(X)), Y, K, L, a.epochs, seed=0)
    export(net, mu, sd, ROOT / "models" / "fuel_volume_conus.json")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "fit_conus.json").write_text(json.dumps(res, indent=1))
    print(f"wrote models/fuel_volume_conus.json  {(time.time() - t0) / 60:.1f} min  peak about {rss():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
