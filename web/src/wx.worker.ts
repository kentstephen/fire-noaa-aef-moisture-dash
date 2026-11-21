import { IcechunkStore } from "icechunk-js";
import * as zarr from "zarrita";

import { BLOCK, blockRowCol, HRRR } from "./wx.grid";

/**
 * Fosberg that builds up and lets go, from the HRRR analysis (dynamical.org, hourly, 3 km), one block at a time.
 *
 * Fosberg's index reads the moisture dead fuel would have if it matched the air at once. Here the moisture follows
 * the air with a time lag instead (the 10 and 100 hour fuel classes of the US fire danger system):
 *     m(t) = target(t) + (m(t-1) - target(t)) * exp(-1 / lag)
 * target is the air's equilibrium moisture (from 2 m temperature and humidity), or 30 %, the top of Fosberg's
 * scale, in an hour with rain. Wind is the mean of the last 24 hours, so the afternoon peak does not pulse. The
 * index is Fosberg's own formula on that moisture and that wind (scripts/weather.py is the same arithmetic).
 *
 * The store's chunks hold 2,160 hours each; the newest one is read, and the one before it too when the newest is
 * younger than SPIN + 3 days. What goes back is the index once a day (00 UTC) per lag; the hourly fields are let go.
 */

const URL_ = "https://dynamical-noaa-hrrr.s3.us-west-2.amazonaws.com/noaa-hrrr-analysis/v0.2.0.icechunk";
const VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m", "precipitation_surface"] as const;
export const LAGS = [10, 100];
/** Days at the start of what is read that are not shown: the moisture is still finding its level. */
export const SPIN = 8;
const CHUNK_T = 2160;
const RAIN_MM = 0.25;

export type WxAsk = { type: "init" } | { type: "block"; id: number; block: number };
export type WxAnswer =
  /** days: the 00 UTC times (ms) the index is given for, oldest first */
  | { type: "ready"; days: number[] }
  /** fos: [day][lag][pixel in block] */
  | { type: "block"; id: number; block: number; fos: Float32Array }
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
  for (let i = SPIN * 24; i < hourMs.length; i++) if (hourMs[i]! % 86_400_000 === 0) dayAt.push(i);
  post({ type: "ready", days: dayAt.map((i) => hourMs[i]!) });
}

async function block(id: number, b: number) {
  const [by, bx] = blockRowCol(b);
  const y0 = by * BLOCK, x0 = bx * BLOCK, h = Math.min(BLOCK, HRRR.h - y0), w = Math.min(BLOCK, HRRR.w - x0);
  const sel = [zarr.slice(t0, t1), zarr.slice(y0, y0 + h), zarr.slice(x0, x0 + w)];
  const [T, RH, U, V, P] = (await Promise.all(VARS.map((v) => zarr.get(arrays[v], sel)))).map((c) => c.data as Float32Array);
  const n = t1 - t0, px = h * w;
  const m = LAGS.map(() => new Float32Array(px).fill(NaN));
  const decay = LAGS.map((l) => Math.exp(-1 / l));
  const hist = new Float32Array(24 * px), sum = new Float32Array(px), seen = new Uint8Array(px);
  const fos = new Float32Array(dayAt.length * LAGS.length * BLOCK * BLOCK).fill(NaN);
  let d = 0;
  for (let i = 0; i < n; i++) {
    const o = i * px, slot = (i % 24) * px;
    for (let p = 0; p < px; p++) {
      const t = T![o + p]!, rh = RH![o + p]!;
      if (t === t && rh === rh) {
        const rain = P![o + p]! * 3600;
        const target = rain >= RAIN_MM ? 30 : emc(rh, t);
        for (let k = 0; k < LAGS.length; k++) m[k]![p] = m[k]![p]! === m[k]![p]! ? target + (m[k]![p]! - target) * decay[k]! : target;
      }
      const u = U![o + p]!, v = V![o + p]!;
      if (u === u && v === v) {
        const mph = Math.hypot(u, v) * 2.2369363;
        if (seen[p]! < 24) seen[p]!++;
        else sum[p]! -= hist[slot + p]!;
        hist[slot + p] = mph;
        sum[p]! += mph;
      }
    }
    if (d < dayAt.length && dayAt[d] === i) {
      for (let k = 0; k < LAGS.length; k++) {
        const at = (d * LAGS.length + k) * BLOCK * BLOCK;
        for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
          const p = r * w + c;
          if (seen[p]) fos[at + r * BLOCK + c] = fosberg(m[k]![p]!, sum[p]! / seen[p]!);
        }
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
