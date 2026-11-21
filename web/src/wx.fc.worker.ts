import { IcechunkStore } from "icechunk-js";
import * as zarr from "zarrita";

import { HRRR } from "./wx.grid";

/**
 * The hour's fire weather as the Fosberg index, from the HRRR 48 hour forecast store (dynamical.org, 3 km), one shard at a
 * time. The store keeps each run in pieces of 300 x 265 pixels (about 900 x 800 km) holding all of the run's hours;
 * the lower 48 is 24 of them.
 *
 * The index (Fosberg 1978) is the 10 m wind times how dry the air would make fine dead fuel (its equilibrium moisture,
 * from 2 m temperature and humidity), 0 to about 100 and past it in the strongest dry wind. It has no term for heat
 * on its own, so a cold, dry wind (Santa Ana, Diablo) reads as high as a hot one. A fitted ignition score that also
 * weighed temperature left such days under the gate (the Camp fire's morning, 8.7 C: Fosberg 43, in that fall's top
 * 4 % of hours); against the 2026 California wildfires the two tell fire hours from other days about equally.
 *
 * A shard's hours are a chain of runs 48 hours apart, each run's first 48 hours, the newest run last (so the chain
 * ends two days ahead of that run's start). After each run two sets of frames go back, oldest run first, so the map
 * fills forward in time while the later runs are still coming:
 *   frames  the index of each hour
 *   slow    the same index sustained: it rises with the hour's index at once and falls back toward it slowly (a
 *           memory of SLOW hours on the way down), given every SLOW_STEP hours. A place that is lifted every
 *           afternoon stays lifted through the nights between; it lets go over a couple of days once the dry
 *           wind stops coming back.
 * Both as bytes: 0 where there is none, 1 to 255 over FOS_LO to FOS_HI.
 */

const URL_ = "https://dynamical-noaa-hrrr.s3.us-west-2.amazonaws.com/noaa-hrrr-forecast-48-hour/v0.1.0.icechunk";
const VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"] as const;
export const SHARD = { w: 300, h: 265 };
/** Hours taken from each run. */
export const RUN_HOURS = 48;
/** The sustained index: how long it holds on the way down (hours), and the hours between its frames. */
const SLOW = 48;
export const SLOW_STEP = 3;
/** The index's range in the bytes (half a point a step). */
export const FOS_LO = 0, FOS_HI = 127;
const byteOf = (z: number) => 1 + Math.round(254 * Math.min(1, Math.max(0, (z - FOS_LO) / (FOS_HI - FOS_LO))));
/** Runs are every 6 hours since this time. */
const ORIGIN = Date.UTC(2018, 6, 13, 12), INIT_STEP = 6 * 3_600_000;

/** end: the start time (ms) of the last run in the chain; null for the newest run in the store. */
export type FcAsk = { type: "init"; runs: number; end: number | null } | { type: "shard"; id: number; row: number; col: number };
export type FcAnswer =
  /** times: the times (ms) of the frames over the whole chain, oldest first; perRun: frames in one run */
  | { type: "ready"; times: number[]; perRun: number }
  /** frames (each hour) and slow (built up, every SLOW_STEP hours): [frame of this run][pixel of the shard, w x h]; run counts from the oldest; last: the chain is done */
  | { type: "run"; id: number; run: number; w: number; h: number; frames: Uint8Array; slow: Uint8Array; last: boolean }
  | { type: "error"; id?: number; message: string };

const post = (a: FcAnswer, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(a, transfer);

let arrays: Record<(typeof VARS)[number], zarr.Array<"float32", zarr.Readable>>;
/** The chain's runs as places on the store's run axis, oldest first. */
let inits: number[] = [];

function emc(rh: number, tC: number) {
  const tF = tC * 1.8 + 32, h = Math.min(100, Math.max(0, rh));
  const m = h < 10 ? 0.03229 + 0.281073 * h - 0.000578 * h * tF : h < 50 ? 2.22749 + 0.160107 * h - 0.014784 * tF : 21.0606 + 0.005565 * h * h - 0.00035 * h * tF - 0.483199 * h;
  return Math.min(30, Math.max(0, m));
}
const fosberg = (m: number, mph: number) => {
  const r = m / 30;
  return ((1 - 2 * r + 1.5 * r * r - 0.5 * r * r * r) * Math.sqrt(1 + mph * mph)) / 0.3002;
};

async function init(runs: number, end: number | null) {
  const store = await IcechunkStore.open(URL_, { withRangeCoalescing: zarr.withRangeCoalescing } as never);
  arrays = {} as typeof arrays;
  await Promise.all(VARS.map(async (v) => {
    const a = await zarr.open(store.resolve(`/${v}`), { kind: "array" });
    if (!a.is("float32")) throw new Error(`${v} is ${a.dtype}`);
    arrays[v] = a;
  }));
  const top = arrays.temperature_2m.shape[0]! - 1;
  const newest = end == null ? top : Math.min(top, Math.round((end - ORIGIN) / INIT_STEP));
  const every = RUN_HOURS / 6;
  inits = Array.from({ length: runs }, (_, k) => newest - (runs - 1 - k) * every).filter((i) => i >= 0);
  const t0 = ORIGIN + inits[0]! * INIT_STEP;
  post({ type: "ready", times: Array.from({ length: inits.length * RUN_HOURS }, (_, f) => t0 + f * 3_600_000), perRun: RUN_HOURS });
}

async function shard(id: number, row: number, col: number) {
  const y0 = row * SHARD.h, x0 = col * SHARD.w, h = Math.min(SHARD.h, HRRR.h - y0), w = Math.min(SHARD.w, HRRR.w - x0);
  const px = h * w;
  const read = (k: number) => Promise.all(VARS.map((v) => zarr.get(arrays[v], [inits[k]!, zarr.slice(0, RUN_HOURS), zarr.slice(y0, y0 + h), zarr.slice(x0, x0 + w)])));
  // the sustained index carries over from run to run
  const built = new Float32Array(px).fill(NaN);
  const keep = Math.exp(-1 / SLOW);
  // the next run is fetched while this one is worked
  let coming = read(0);
  for (let k = 0; k < inits.length; k++) {
    const got = await coming;
    if (k + 1 < inits.length) coming = read(k + 1);
    const [T, RH, U, V] = got.map((c) => c.data as Float32Array);
    const frames = new Uint8Array(RUN_HOURS * px), slow = new Uint8Array((RUN_HOURS / SLOW_STEP) * px);
    for (let i = 0; i < RUN_HOURS; i++) {
      const o = i * px, sa = i % SLOW_STEP === 0 ? (i / SLOW_STEP) * px : -1;
      for (let p = 0; p < px; p++) {
        const t = T![o + p]!, rh = RH![o + p]!, u = U![o + p]!, v = V![o + p]!;
        if (t === t && rh === rh && u === u && v === v) {
          const z = fosberg(emc(rh, t), Math.hypot(u, v) * 2.2369363);
          frames[o + p] = byteOf(z);
          // up at once with the hour, down slowly: a place that lights each afternoon stays lit through the night
          built[p] = built[p]! > z ? z + (built[p]! - z) * keep : z;
        }
        if (sa >= 0 && built[p]! === built[p]!) slow[sa + p] = byteOf(built[p]!);
      }
    }
    post({ type: "run", id, run: k, w, h, frames, slow, last: k === inits.length - 1 }, [frames.buffer, slow.buffer]);
  }
}

self.onmessage = (ev: MessageEvent<FcAsk>) => {
  const d = ev.data;
  if (d.type === "init") init(d.runs, d.end).catch((e) => post({ type: "error", message: String(e?.message ?? e) }));
  else shard(d.id, d.row, d.col).catch((e) => post({ type: "error", id: d.id, message: String(e?.message ?? e) }));
};
