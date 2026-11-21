import { IcechunkStore } from "icechunk-js";
import * as zarr from "zarrita";

import { BLOCK, blockRowCol, HRRR } from "./wx.grid";

/**
 * Fosberg that builds up and lets go, from the HRRR analysis (dynamical.org, hourly, 3 km), one block at a time.
 *
 * Fosberg's index reads the moisture dead fuel would have if it matched the air at once, so it swings from day to
 * night. Here the moisture follows the air with a time lag instead (the 100 hour fuel class of the US fire danger
 * system):
 *     m(t) = target(t) + (m(t-1) - target(t)) * exp(-1 / LAG)
 * target is the air's equilibrium moisture (from 2 m temperature and humidity), or 30 %, the top of Fosberg's
 * scale, in an hour with rain. Wind is the mean of the last WIND hours. The index is Fosberg's own formula on that
 * moisture and that wind (scripts/weather.py is the same arithmetic). What is left of the daily swing is small;
 * what shows is dry wind adding up over days and rain or damp air taking it back.
 *
 * The store's chunks hold 2,160 hours each; the newest one is read, and the one before it too when the newest is
 * younger than SPIN + 3 days. What goes back is the index every STEP hours as bytes (WX_SCALE steps per point of
 * Fosberg, 0 where there is none); the hourly fields are let go.
 */

const URL_ = "https://dynamical-noaa-hrrr.s3.us-west-2.amazonaws.com/noaa-hrrr-analysis/v0.2.0.icechunk";
const VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m", "precipitation_surface"] as const;
/** Hours: the moisture lag, the span of the wind mean, and the time between the frames given back. */
const LAG = 100;
const WIND = 24;
export const STEP = 6;
const WX_SCALE = 3;
/** Days at the start of what is read that are not shown: the moisture is still finding its level. */
export const SPIN = 8;
const CHUNK_T = 2160;
const RAIN_MM = 0.25;

export type WxAsk = { type: "init" } | { type: "block"; id: number; block: number };
export type WxAnswer =
  /** times: the times (ms) of the frames, oldest first */
  | { type: "ready"; times: number[] }
  /** fos: [frame][pixel in block], bytes */
  | { type: "block"; id: number; block: number; fos: Uint8Array }
  | { type: "error"; id?: number; message: string };

const post = (a: WxAnswer, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(a, transfer);

let arrays: Record<(typeof VARS)[number], zarr.Array<"float32", zarr.Readable>>;
let t0 = 0, t1 = 0;
let hourMs: Float64Array;
let dayAt: number[] = [];

function emc(rh: number, tC: number) {
  const tF = tC * 1.8 + 32, h = Math.min(100, Math.max(0, rh));
  const m = h < 10 ? 0.03229 + 0.281073 * h - 0.000578 * h * tF : h < 50 ? 2.22749 + 0.160107 * h - 0.014784 * tF : 21.0606 + 0.005565 * h * h - 0.00035 * h * tF - 0.483199 * h;
  return Math.min(30, Math.max(0, m));
}
const fosberg = (m: number, mph: number) => {
  const r = m / 30;
  return ((1 - 2 * r + 1.5 * r * r - 0.5 * r * r * r) * Math.sqrt(1 + mph * mph)) / 0.3002;
};

async function init() {
  const store = await IcechunkStore.open(URL_, { withRangeCoalescing: zarr.withRangeCoalescing } as never);
  arrays = {} as typeof arrays;
  await Promise.all(VARS.map(async (v) => {
    const a = await zarr.open(store.resolve(`/${v}`), { kind: "array" });
    if (!a.is("float32")) throw new Error(`${v} is ${a.dtype}`);
    arrays[v] = a;
  }));
  const time = await zarr.open(store.resolve("/time"), { kind: "array" });
  t1 = arrays.temperature_2m.shape[0]!;
  const last = Math.floor((t1 - 1) / CHUNK_T);
  t0 = (t1 - last * CHUNK_T < (SPIN + 3) * 24 ? last - 1 : last) * CHUNK_T;
  const tv = (await zarr.get(time, [zarr.slice(t0, t1)])).data as ArrayLike<number | bigint>;
  // seconds since 1970 in the store, whatever its integer width
  hourMs = Float64Array.from(tv as ArrayLike<number>, (v) => Number(v) * 1000);
  dayAt = [];
  for (let i = SPIN * 24; i < hourMs.length; i++) if (hourMs[i]! % (STEP * 3_600_000) === 0) dayAt.push(i);
  post({ type: "ready", times: dayAt.map((i) => hourMs[i]!) });
}

async function block(id: number, b: number) {
  const [by, bx] = blockRowCol(b);
  const y0 = by * BLOCK, x0 = bx * BLOCK, h = Math.min(BLOCK, HRRR.h - y0), w = Math.min(BLOCK, HRRR.w - x0);
  const sel = [zarr.slice(t0, t1), zarr.slice(y0, y0 + h), zarr.slice(x0, x0 + w)];
  const [T, RH, U, V, P] = (await Promise.all(VARS.map((v) => zarr.get(arrays[v], sel)))).map((c) => c.data as Float32Array);
  const n = t1 - t0, px = h * w;
  const m = new Float32Array(px).fill(NaN);
  const decay = Math.exp(-1 / LAG);
  const hist = new Float32Array(WIND * px), sum = new Float32Array(px), seen = new Uint8Array(px);
  const fos = new Uint8Array(dayAt.length * BLOCK * BLOCK);
  let d = 0;
  for (let i = 0; i < n; i++) {
    const o = i * px, slot = (i % WIND) * px;
    for (let p = 0; p < px; p++) {
      const t = T![o + p]!, rh = RH![o + p]!;
      if (t === t && rh === rh) {
        const rain = P![o + p]! * 3600;
        const target = rain >= RAIN_MM ? 30 : emc(rh, t);
        m[p] = m[p]! === m[p]! ? target + (m[p]! - target) * decay : target;
      }
      const u = U![o + p]!, v = V![o + p]!;
      if (u === u && v === v) {
        const mph = Math.hypot(u, v) * 2.2369363;
        if (seen[p]! < WIND) seen[p]!++;
        else sum[p]! -= hist[slot + p]!;
        hist[slot + p] = mph;
        sum[p]! += mph;
      }
    }
    if (d < dayAt.length && dayAt[d] === i) {
      const at = d * BLOCK * BLOCK;
      for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
        const p = r * w + c;
        if (seen[p] && m[p]! === m[p]!) fos[at + r * BLOCK + c] = Math.min(255, Math.max(1, Math.round(fosberg(m[p]!, sum[p]! / seen[p]!) * WX_SCALE)));
      }
      d++;
    }
  }
  post({ type: "block", id, block: b, fos }, [fos.buffer]);
}

self.onmessage = (ev: MessageEvent<WxAsk>) => {
  const d = ev.data;
  if (d.type === "init") init().catch((e) => post({ type: "error", message: String(e?.message ?? e) }));
  else block(d.id, d.block).catch((e) => post({ type: "error", id: d.id, message: String(e?.message ?? e) }));
};
