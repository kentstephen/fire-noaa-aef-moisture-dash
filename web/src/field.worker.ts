import { cellToChildren, lonLatToCell } from "a5-js";
import { tableFromIPC } from "apache-arrow";
import initWasm, { ParquetFile } from "parquet-wasm/esm";
import wasmUrl from "parquet-wasm/esm/parquet_wasm_bg.wasm?url";

import { volT } from "./ramp";

/**
 * The fuel as map tiles, made here from the A5 cells. The cells come from Parquet (parquet-wasm, HTTP range requests,
 * one row group at a time); a tile asks, for each of its texels, which cell the texel's center is in and takes that
 * cell's fuel. The cells are picked about as wide as a texel, so on the map a cell is a point of the picture and its
 * outline is never drawn. Nothing is cut into triangles and no corners are computed.
 *
 * A tile is TILE texels a side plus one texel of its neighbors all round (so the blend between texels runs across
 * tile edges), four bytes a texel: the fuel on the ramp's scale, twice unused, and whether there is ground there.
 */

export const TILE = 256;
export const SIDE = TILE + 2;

export type FieldAsk =
  | { type: "init"; base: string }
  /** The tiles wanted now, nearest first; anything asked before and not started is dropped. */
  | { type: "want"; jobs: { z: number; x: number; y: number; res: number }[] };
export type FieldAnswer =
  /** The resolutions the files hold. */
  | { type: "index"; res: number[] }
  | { type: "tile"; z: number; x: number; y: number; res: number; data: Uint8Array | null }
  | { type: "error"; z?: number; x?: number; y?: number; message: string };

const post = (a: FieldAnswer, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(a, transfer);

let base = "";
let ready: Promise<unknown> | null = null;
const files = new Map<number, Promise<ParquetFile>>();
const open = (name: string) => ParquetFile.fromUrl(new URL(name, base).href);
const fileOf = (res: number) => {
  let f = files.get(res);
  if (!f) files.set(res, (f = open(`data/fuel_r${res}.parquet`)));
  return f;
};
/** Per resolution: each row group's first and last cell ID, in order. */
const index = new Map<number, { lo: BigUint64Array; hi: BigUint64Array }>();

/** A row group in hand: its cell IDs as pairs of 32 bit halves (low, high), and each cell's place on the ramp. */
type Group = { ids: Uint32Array; t: Uint8Array; seen: number };
const groups = new Map<string, Promise<Group>>();
const MAX_GROUPS = 120;
let tick = 0;

function group(res: number, rg: number) {
  const key = `${res}/${rg}`;
  let g = groups.get(key);
  if (!g) {
    g = (async () => {
      const file = await fileOf(res);
      const t = tableFromIPC((await file.read({ rowGroups: [rg], columns: ["cell", "vol"] })).intoIPCStream());
      const cells = (t.getChild("cell")!.toArray() as BigUint64Array).slice();
      const vol = t.getChild("vol")!.toArray() as Float32Array;
      const tt = new Uint8Array(vol.length);
      for (let i = 0; i < vol.length; i++) tt[i] = Math.round(255 * volT(vol[i]!));
      return { ids: new Uint32Array(cells.buffer), t: tt, seen: tick };
    })();
    groups.set(key, g);
    g.catch(() => groups.delete(key));
    if (groups.size > MAX_GROUPS) {
      // let go of the ones used longest ago (the order of first use is close enough)
      for (const k of groups.keys()) {
        if (groups.size <= MAX_GROUPS) break;
        groups.delete(k);
      }
    }
  }
  return g;
}

const EARTH_M2 = 5.1007e14;
const cellM = (res: number) => Math.sqrt(EARTH_M2 / (60 * 4 ** (res - 1)));

/** A cell's first and last descendant IDs at a finer resolution (children are in ID order). */
const spans = new Map<string, [bigint, bigint]>();
function span(cell: bigint, from: number, to: number): [bigint, bigint] {
  const k = `${cell}>${to}`;
  let s = spans.get(k);
  if (!s) {
    let lo = cell, hi = cell;
    for (let r = from; r < to; r++) {
      const a = cellToChildren(lo), b = cellToChildren(hi);
      lo = a.reduce((m, c) => (c < m ? c : m));
      hi = b.reduce((m, c) => (c > m ? c : m));
    }
    if (spans.size > 20_000) spans.clear();
    spans.set(k, (s = [lo, hi]));
  }
  return s;
}

/** The row groups of a resolution that hold cells inside a box (west, south, east, north). */
function rowGroups(res: number, bbox: [number, number, number, number]): number[] {
  const ix = index.get(res)!;
  if (ix.lo.length <= 2) return [...ix.lo.keys()];
  // cover the box with cells five resolutions coarser (32 times wider), found by sampling points closer than those cells
  const cover = Math.max(1, res - 5);
  const [w, s, e, n] = bbox;
  const step = cellM(cover) / 4 / 111_000;
  const nx = Math.min(200, Math.max(2, Math.ceil((e - w) / (step / Math.cos((((s + n) / 2) * Math.PI) / 180)))));
  const ny = Math.min(200, Math.max(2, Math.ceil((n - s) / step)));
  const parents = new Set<bigint>();
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i <= nx; i++) parents.add(lonLatToCell([w + ((e - w) * i) / nx, s + ((n - s) * j) / ny] as never, cover));
  }
  const out = new Set<number>();
  for (const p of parents) {
    const [lo, hi] = span(p, cover, res);
    // the first row group ending at or after lo, then on while they start at or before hi
    let a = 0, b = ix.hi.length;
    while (a < b) {
      const m = (a + b) >> 1;
      if (ix.hi[m]! < lo) a = m + 1;
      else b = m;
    }
    for (let g = a; g < ix.lo.length && ix.lo[g]! <= hi; g++) out.add(g);
  }
  return [...out].sort((x, y) => x - y);
}

const lonOf = (fx: number) => fx * 360 - 180;
const latOf = (fy: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * fy))) * 180) / Math.PI;

async function tile(z: number, x: number, y: number, res: number) {
  const n = 2 ** z, px = 1 / (n * TILE);
  const fx0 = x / n - px, fy0 = y / n - px;
  const rgs = index.has(res) ? rowGroups(res, [lonOf(fx0), latOf(fy0 + SIDE * px), lonOf(fx0 + SIDE * px), latOf(fy0)]) : [];
  if (!rgs.length) return post({ type: "tile", z, x, y, res, data: null });
  tick++;
  const gs = await Promise.all(rgs.map((rg) => group(res, rg)));
  // each group's first and last ID as halves, for finding a texel's group
  const m = gs.length;
  const loH = new Float64Array(m), loL = new Float64Array(m), hiH = new Float64Array(m), hiL = new Float64Array(m);
  gs.forEach((g, i) => {
    const e = g.ids.length - 2;
    loL[i] = g.ids[0]!, loH[i] = g.ids[1]!, hiL[i] = g.ids[e]!, hiH[i] = g.ids[e + 1]!;
    g.seen = tick;
  });
  const data = new Uint8Array(SIDE * SIDE * 4);
  let any = false, gi = 0, lastH = -1, lastL = -1, lastT = -1;
  const pt = [0, 0];
  for (let j = 0; j < SIDE; j++) {
    pt[1] = latOf(fy0 + (j + 0.5) * px);
    for (let i = 0; i < SIDE; i++) {
      pt[0] = lonOf(fx0 + (i + 0.5) * px);
      const c = lonLatToCell(pt as never, res);
      const h = Number(c >> 32n), l = Number(c & 0xffffffffn);
      let t = -1;
      if (h === lastH && l === lastL) t = lastT;
      else {
        // the group: the one the last texel was in, most of the time
        if (!(gi < m && (h > loH[gi]! || (h === loH[gi]! && l >= loL[gi]!)) && (h < hiH[gi]! || (h === hiH[gi]! && l <= hiL[gi]!)))) {
          gi = 0;
          while (gi < m && (h > hiH[gi]! || (h === hiH[gi]! && l > hiL[gi]!))) gi++;
        }
        if (gi < m) {
          const ids = gs[gi]!.ids;
          let a = 0, b = ids.length / 2 - 1;
          while (a <= b) {
            const k = (a + b) >> 1, kh = ids[2 * k + 1]!, kl = ids[2 * k]!;
            if (kh === h && kl === l) {
              t = gs[gi]!.t[k]!;
              break;
            }
            if (kh < h || (kh === h && kl < l)) a = k + 1;
            else b = k - 1;
          }
        }
        lastH = h, lastL = l, lastT = t;
      }
      if (t >= 0) {
        const o = 4 * (j * SIDE + i);
        data[o] = t;
        data[o + 3] = 255;
        any = true;
      }
    }
  }
  if (!any) return post({ type: "tile", z, x, y, res, data: null });
  post({ type: "tile", z, x, y, res, data }, [data.buffer]);
}

let queue: { z: number; x: number; y: number; res: number }[] = [];
let busy = 0;
/** Tiles under way at once: their reads overlap, their texel loops take turns. */
const PARALLEL = 3;
function pump() {
  while (busy < PARALLEL && queue.length) {
    const j = queue.shift()!;
    busy++;
    tile(j.z, j.x, j.y, j.res)
      .catch((e) => post({ type: "error", z: j.z, x: j.x, y: j.y, message: String(e?.message ?? e) }))
      .finally(() => {
        busy--;
        pump();
      });
  }
}

self.onmessage = async (ev: MessageEvent<FieldAsk>) => {
  const d = ev.data;
  try {
    if (d.type === "init") {
      base = d.base;
      ready = (async () => {
        await initWasm({ module_or_path: wasmUrl });
        const t = tableFromIPC((await (await open("data/fuel_index.parquet")).read()).intoIPCStream());
        const res = t.getChild("res")!.toArray() as Int8Array;
        const lo = t.getChild("lo")!.toArray() as BigUint64Array, hi = t.getChild("hi")!.toArray() as BigUint64Array;
        for (const r of new Set(res)) {
          const at: number[] = [];
          for (let i = 0; i < res.length; i++) if (res[i] === r) at.push(i);
          index.set(r, { lo: BigUint64Array.from(at, (i) => lo[i]!), hi: BigUint64Array.from(at, (i) => hi[i]!) });
        }
        post({ type: "index", res: [...index.keys()].sort((a, b) => a - b) });
      })();
      await ready;
    } else if (d.type === "want") {
      await ready;
      queue = d.jobs;
      pump();
    }
  } catch (e) {
    post({ type: "error", message: String((e as Error)?.message ?? e) });
  }
};
