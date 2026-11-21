import { cellToBoundary, cellToLonLat } from "a5-js";
import { tableFromIPC } from "apache-arrow";
import initWasm, { ParquetFile } from "parquet-wasm/esm";
import wasmUrl from "parquet-wasm/esm/parquet_wasm_bg.wasm?url";

import { rampInto, volT } from "./ramp";
import { hrrrXY } from "./wx.grid";

/**
 * Reads the cells from Parquet (parquet-wasm, HTTP range requests, one row group at a time) and turns each cell
 * into its pentagon and its color, off the main thread. The page asks for row groups by (resolution, index);
 * what comes back is typed arrays, handed over without a copy.
 */

export type Ask =
  | { type: "init"; base: string }
  /** The row groups wanted now, nearest first; anything asked before and not started is dropped. */
  | { type: "want"; jobs: { res: number; rg: number }[] };
export type Answer =
  | { type: "index"; res: Int8Array; rg: Int32Array; lo: BigUint64Array; hi: BigUint64Array }
  | { type: "group"; res: number; rg: number; cells: BigUint64Array; vol: Float32Array; n: Int32Array; coords: Float64Array; rings: Int32Array; tris: Uint32Array; colors: Uint8Array; wx: Float32Array }
  | { type: "error"; res?: number; rg?: number; message: string };

const post = (a: Answer, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(a, transfer);

let base = "";
let ready: Promise<unknown> | null = null;
const files = new Map<number, Promise<ParquetFile>>();
const open = (name: string) => ParquetFile.fromUrl(new URL(name, base).href);
const fileOf = (res: number) => {
  let f = files.get(res);
  if (!f) files.set(res, (f = open(`data/fuel_r${res}.parquet`)));
  return f;
};

let queue: { res: number; rg: number }[] = [];
const busy = new Set<string>();
const PARALLEL = 4;

async function readGroup(res: number, rg: number) {
  const file = await fileOf(res);
  const t = tableFromIPC((await file.read({ rowGroups: [rg] })).intoIPCStream());
  const cells = t.getChild("cell")!.toArray() as BigUint64Array;
  const vol = t.getChild("vol")!.toArray() as Float32Array;
  const n = t.getChild("n")!.toArray() as Int32Array;
  const count = cells.length;
  // a pentagon is five corners and the first again; a cell with more is allowed for
  let coords = new Float64Array(count * 12);
  const rings = new Int32Array(count + 1);
  const colors = new Uint8Array(count * 4);
  // each cell's center on the weather grid (HRRR, 3 km): column, row
  const wx = new Float32Array(count * 2);
  // each pentagon as a fan of three triangles from its first corner (the cells are convex), so the page has no cutting to do
  let tris = new Uint32Array(count * 9);
  let q = 0;
  let p = 0;
  for (let i = 0; i < count; i++) {
    const ring = cellToBoundary(cells[i]!, { closedRing: true, segments: 1 });
    if ((p + ring.length) * 2 > coords.length) {
      const grown = new Float64Array(coords.length * 2);
      grown.set(coords);
      coords = grown;
    }
    const corners = ring.length - 1;
    if (q + 3 * (corners - 2) > tris.length) {
      const grown = new Uint32Array(tris.length * 2);
      grown.set(tris);
      tris = grown;
    }
    for (let k = 1; k < corners - 1; k++) {
      tris[q++] = p;
      tris[q++] = p + k;
      tris[q++] = p + k + 1;
    }
    for (const pt of ring) {
      coords[2 * p] = pt[0];
      coords[2 * p + 1] = pt[1];
      p++;
    }
    rings[i + 1] = p;
    rampInto(colors, 4 * i, volT(vol[i]!));
    const ctr = cellToLonLat(cells[i]!);
    hrrrXY(ctr[0], ctr[1], wx, 2 * i);
  }
  if (p * 2 !== coords.length) coords = coords.slice(0, p * 2);
  if (q !== tris.length) tris = tris.slice(0, q);
  // copies of the columns: the table's own buffers are views into the IPC bytes
  const c2 = cells.slice(), v2 = vol.slice(), n2 = n.slice();
  post({ type: "group", res, rg, cells: c2, vol: v2, n: n2, coords, rings, tris, colors, wx }, [c2.buffer, v2.buffer, n2.buffer, coords.buffer, rings.buffer, tris.buffer, colors.buffer, wx.buffer]);
}

function pump() {
  while (busy.size < PARALLEL && queue.length) {
    const job = queue.shift()!;
    const key = `${job.res}/${job.rg}`;
    if (busy.has(key)) continue;
    busy.add(key);
    readGroup(job.res, job.rg)
      .catch((e) => post({ type: "error", res: job.res, rg: job.rg, message: String(e?.message ?? e) }))
      .finally(() => {
        busy.delete(key);
        pump();
      });
  }
}

self.onmessage = async (ev: MessageEvent<Ask>) => {
  const d = ev.data;
  try {
    if (d.type === "init") {
      base = d.base;
      ready = initWasm({ module_or_path: wasmUrl });
      await ready;
      const t = tableFromIPC((await (await open("data/fuel_index.parquet")).read()).intoIPCStream());
      const col = (k: string) => t.getChild(k)!.toArray();
      const res = (col("res") as Int8Array).slice(), rg = (col("rg") as Int32Array).slice();
      const lo = (col("lo") as BigUint64Array).slice(), hi = (col("hi") as BigUint64Array).slice();
      post({ type: "index", res, rg, lo, hi }, [res.buffer, rg.buffer, lo.buffer, hi.buffer]);
    } else if (d.type === "want") {
      await ready;
      queue = d.jobs;
      pump();
    }
  } catch (e) {
    post({ type: "error", message: String((e as Error)?.message ?? e) });
  }
};
