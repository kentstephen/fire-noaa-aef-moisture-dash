import type { Device } from "@luma.gl/core";

import { setWxTex } from "./risk";
import type { FcAnswer, FcAsk } from "./wx.fc.worker";
import { FOS_HI, FOS_LO, RUN_HOURS, SHARD, SLOW_STEP } from "./wx.fc.worker";
import FcWorker from "./wx.fc.worker.ts?worker";
import { HRRR, hrrrXY } from "./wx.grid";

/**
 * The weather on the map: the hour's Fosberg index, and the same index built up over days, read in the browser
 * from the HRRR forecast store (wx.fc.worker.ts), only the shards under the view. Each shard's runs arrive oldest
 * first and are put into two textures (a layer per frame) that the map's shader reads at every screen pixel, so the
 * picture fills forward in time as the data comes; nothing here colors anything. Frames a shard does not have yet
 * hold its newest one.
 */

/** Runs in the chain, 48 hours each: one behind the newest (two days), and the newest, which ends two days ahead. */
const RUNS = 2;
const COLS = Math.ceil(HRRR.w / SHARD.w), ROWS = Math.ceil(HRRR.h / SHARD.h);
const WORKERS = 4;
const LINEAR = { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" } as const;

/** A shard's frames so far: [frame][pixel], and how many frames are in. */
type Shard = { w: number; h: number; data: Uint8Array; slow: Uint8Array; frames: number; done: boolean };
/** Built-up frames in a run. */
const SLOW_RUN = RUN_HOURS / SLOW_STEP;

export function createWeather(onChange: () => void, onError: (m: string) => void) {
  const workers: { w: Worker; busy: number; ready: boolean }[] = [];
  const shards = new Map<number, Shard>();
  const failed = new Set<number>();
  const flying = new Map<number, number>();
  let times: number[] = [];
  let perRun = 0;
  let queue: number[] = [];
  let want: number[] = [];
  let nextId = 1;
  /** bumps when the frames under the view change; the texture is made again when it is behind */
  let rev = 0, texRev = -1, texAt = 0;
  /** end: the start (ms) of the last run of the window read; null is the newest run, which ends two days ahead of now. */
  const st: { on: boolean; end: number | null } = { on: true, end: null };

  function pump() {
    while (queue.length) {
      const w = workers.find((x) => x.ready && !x.busy);
      if (!w) return;
      const s = queue.shift()!;
      if (shards.get(s)?.done || failed.has(s) || [...flying.values()].includes(s)) continue;
      const id = nextId++;
      flying.set(id, s);
      w.busy++;
      w.w.postMessage({ type: "shard", id, row: Math.floor(s / COLS), col: s % COLS } satisfies FcAsk);
    }
  }

  function start() {
    if (workers.length) return;
    for (let i = 0; i < WORKERS; i++) {
      const ent = { w: new FcWorker(), busy: 0, ready: false };
      ent.w.onmessage = (ev: MessageEvent<FcAnswer>) => {
        const d = ev.data;
        if (d.type === "ready") {
          ent.ready = true;
          if (!times.length) (times = d.times), (perRun = d.perRun);
        } else if (d.type === "run") {
          const s = flying.get(d.id)!;
          let sh = shards.get(s);
          if (!sh || d.run === 0) shards.set(s, (sh = { w: d.w, h: d.h, data: new Uint8Array(times.length * d.w * d.h), slow: new Uint8Array((times.length / SLOW_STEP) * d.w * d.h), frames: 0, done: false }));
          sh.data.set(d.frames, d.run * perRun * d.w * d.h);
          sh.slow.set(d.slow, d.run * SLOW_RUN * d.w * d.h);
          sh.frames = (d.run + 1) * perRun;
          sh.done = d.last;
          if (d.last) {
            flying.delete(d.id);
            ent.busy--;
          }
        } else {
          if (d.id != null) {
            const s = flying.get(d.id);
            flying.delete(d.id);
            ent.busy--;
            if (s != null) failed.add(s);
          }
          onError(`weather: ${d.message}`);
        }
        rev++;
        pump();
        onChange();
      };
      ent.w.onerror = (e) => onError(`weather worker: ${e.message || "failed to start"} ${e.filename ? `(${e.filename.split("/").pop()}:${e.lineno})` : ""}`);
      ent.w.postMessage({ type: "init", runs: RUNS, end: st.end } satisfies FcAsk);
      workers.push(ent);
    }
  }

  /** The byte at a pixel and frame (a frame not in yet reads the newest one that is), 0 where the shard is not in hand. */
  function byteAt(col: number, row: number, frame: number, builtUp: boolean) {
    if (col < 0 || row < 0 || col >= HRRR.w || row >= HRRR.h) return 0;
    const sh = shards.get(Math.floor(row / SHARD.h) * COLS + Math.floor(col / SHARD.w));
    if (!sh?.frames) return 0;
    const at = (row % SHARD.h) * sh.w + (col % SHARD.w);
    return builtUp ? sh.slow[Math.min(frame, sh.frames / SLOW_STEP - 1) * sh.w * sh.h + at]! : sh.data[Math.min(frame, sh.frames - 1) * sh.w * sh.h + at]!;
  }
  const xy = new Float32Array(2);

  return {
    st,
    /** The times (ms) of the frames, oldest first; empty until the store is open. The last two days are ahead of the newest run. */
    get times() {
      return times;
    },
    set(patch: Partial<typeof st>) {
      if ("end" in patch && patch.end !== st.end) {
        // another window: everything read so far is let go and the workers start over on it
        for (const w of workers) w.w.terminate();
        workers.length = 0;
        shards.clear();
        failed.clear();
        flying.clear();
        times = [];
        queue = want.slice();
      }
      Object.assign(st, patch);
      if (st.on) start();
      rev++;
      pump();
      onChange();
    },
    /** The view (west, south, east, north): the shards under it are read, from its middle outward. */
    view(bbox: [number, number, number, number]) {
      const [w, s, e, n] = bbox, found = new Set<number>();
      const K = 48;
      for (let j = 0; j <= K; j++) {
        for (let i = 0; i <= K; i++) {
          hrrrXY(w + ((e - w) * i) / K, s + ((n - s) * j) / K, xy, 0);
          if (xy[0]! === xy[0]!) found.add(Math.floor((xy[1]! + 0.5) / SHARD.h) * COLS + Math.floor((xy[0]! + 0.5) / SHARD.w));
        }
      }
      hrrrXY((w + e) / 2, (s + n) / 2, xy, 0);
      const mc = (xy[0]! === xy[0]! ? xy[0]! : HRRR.w / 2) / SHARD.w - 0.5, mr = (xy[1]! === xy[1]! ? xy[1]! : HRRR.h / 2) / SHARD.h - 0.5;
      const d2 = (s: number) => (Math.floor(s / COLS) - mr) ** 2 + ((s % COLS) - mc) ** 2;
      const next = [...found].filter((s) => s >= 0 && s < COLS * ROWS).sort((a, b) => d2(a) - d2(b));
      if (next.length !== want.length || next.some((b, i) => b !== want[i])) rev++;
      want = next;
      if (!st.on) return;
      start();
      queue = want.filter((s) => !shards.get(s)?.done && !failed.has(s));
      pump();
    },
    /** Frames in hand under the view, of all there will be. */
    progress() {
      let got = 0;
      for (const s of want) got += failed.has(s) ? times.length : (shards.get(s)?.frames ?? 0);
      const total = want.length * times.length;
      return { got: perRun ? got / perRun : 0, total: perRun ? total / perRun : 0, done: !st.on || (times.length > 0 && got === total) };
    },
    /**
     * Put the frames in hand under the view into the shader's texture, when they have changed (at most twice a
     * second while they are arriving). One byte per pixel and frame, plus a mask of the pixels in hand.
     */
    upload(device: Device) {
      if (texRev === rev || !times.length) return;
      const now = performance.now();
      if (now - texAt < 500) return void setTimeout(onChange, 500 - (now - texAt));
      texRev = rev;
      texAt = now;
      const have = want.filter((s) => shards.get(s)?.frames);
      if (!st.on || !have.length) return setWxTex(null);
      let r0 = Infinity, r1 = -1, c0 = Infinity, c1 = -1;
      for (const s of have) {
        const r = Math.floor(s / COLS), c = s % COLS;
        r0 = Math.min(r0, r), r1 = Math.max(r1, r), c0 = Math.min(c0, c), c1 = Math.max(c1, c);
      }
      const w = Math.min(HRRR.w, (c1 + 1) * SHARD.w) - c0 * SHARD.w, h = Math.min(HRRR.h, (r1 + 1) * SHARD.h) - r0 * SHARD.h, frames = times.length;
      const nSlow = frames / SLOW_STEP;
      const data = new Uint8Array(w * h * frames), mask = new Uint8Array(w * h), slow = new Uint8Array(w * h * nSlow);
      for (const s of have) {
        const sh = shards.get(s)!, x = ((s % COLS) - c0) * SHARD.w, y = (Math.floor(s / COLS) - r0) * SHARD.h, per = sh.w * sh.h;
        for (let k = 0; k < frames; k++) {
          const from = Math.min(k, sh.frames - 1) * per;
          for (let j = 0; j < sh.h; j++) data.set(sh.data.subarray(from + j * sh.w, from + (j + 1) * sh.w), k * w * h + (y + j) * w + x);
        }
        for (let k = 0; k < nSlow; k++) {
          const from = Math.min(k, sh.frames / SLOW_STEP - 1) * per;
          for (let j = 0; j < sh.h; j++) slow.set(sh.slow.subarray(from + j * sh.w, from + (j + 1) * sh.w), k * w * h + (y + j) * w + x);
        }
        for (let j = 0; j < sh.h; j++) for (let i = 0; i < sh.w; i++) if (sh.data[j * sh.w + i]) mask[(y + j) * w + x + i] = 255;
      }
      setWxTex({
        tex: device.createTexture({ dimension: "2d-array", width: w, height: h, depth: frames, format: "r8unorm", data, mipLevels: 1, sampler: LINEAR }),
        slow: device.createTexture({ dimension: "2d-array", width: w, height: h, depth: nSlow, format: "r8unorm", data: slow, mipLevels: 1, sampler: LINEAR }),
        slowStep: SLOW_STEP,
        mask: device.createTexture({ width: w, height: h, format: "r8unorm", data: mask, mipLevels: 1, sampler: LINEAR }),
        col0: c0 * SHARD.w,
        row0: r0 * SHARD.h,
        w,
        h,
        frames,
      });
    },
    /**
     * The Fosberg index at a place and an hour (fractions blend the two hours around it), each hour's or the built-up
     * one; NaN where not in hand.
     */
    value(lon: number, lat: number, frame: number, builtUp: boolean): number {
      hrrrXY(lon, lat, xy, 0);
      if (xy[0]! !== xy[0]! || !times.length) return NaN;
      const n = builtUp ? times.length / SLOW_STEP : times.length;
      const f = Math.min(n - 1, Math.max(0, builtUp ? frame / SLOW_STEP : frame)), f0 = Math.floor(f), f1 = Math.min(n - 1, f0 + 1), u = f - f0;
      const c0 = Math.floor(xy[0]!), r0 = Math.floor(xy[1]!), fc = xy[0]! - c0, fr = xy[1]! - r0;
      let acc = 0, wsum = 0;
      for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
        const a = byteAt(c0 + i, r0 + j, f0, builtUp), b = byteAt(c0 + i, r0 + j, f1, builtUp);
        if (!a || !b) continue;
        const w = (i ? fc : 1 - fc) * (j ? fr : 1 - fr);
        acc += w * (a + (b - a) * u);
        wsum += w;
      }
      return wsum > 0.01 ? FOS_LO + ((acc / wsum - 1) / 254) * (FOS_HI - FOS_LO) : NaN;
    },
  };
}
export type Weather = ReturnType<typeof createWeather>;
