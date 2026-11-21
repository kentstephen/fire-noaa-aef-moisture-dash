import { lonLatToCell } from "a5-js";
import type { RecordBatch } from "apache-arrow";

import type { FineAnswer, FineAsk, Head } from "./fine.worker";
import FineWorker from "./fine.worker.ts?worker";
import { cellLayer, colorData, polygonBatch } from "./geo";
import { volT } from "./ramp";

/**
 * Zoomed in past the baked cells: the real 10 m embeddings of the view are read and scored in the browser
 * (fine.worker.ts) and drawn as finer A5 cells over the fuel tiles, shaded the same way. Nothing is kept on disk.
 */

/** From this zoom the embeddings are read: a full window is about 50 to 120 chunks of 2.6 MB. */
export const FINE_ZOOM = 12.5;
/**
 * The cells made: resolution 18 (about 22 m, six 10 m pixels each), and from FINEST_ZOOM resolution 19 (about 11 m,
 * the size of a pixel, which is as far as the embeddings go).
 */
const FINEST_ZOOM = 14;
export const fineRes = (zoom: number) => (zoom >= FINEST_ZOOM ? 19 : 18);
/** AlphaEarth on Source Cooperative, through its gateway; the year's place on the time axis (2017 is 0). */
const ROOT = "https://data.source.coop/tge-labs/aef-mosaic";
const YEAR = 2025;
const RES = 8.983111749910169e-5, X0 = -180, Y0 = 83.68570533713473, CHUNK = 256;
/** Chunks kept (1.5 MB each as cells at resolution 18, 6 MB at 19) before the ones seen longest ago are let go. */
const MAX_CHUNKS = 120;
/** Reads in flight per worker: the wait is the network, the scoring is one at a time anyway. */
const LANES = 3;

type Chunk = {
  cells: BigUint64Array;
  sum: Float32Array;
  n: Uint16Array;
  own: Int32Array;
  tris: Uint32Array;
  batch: RecordBatch;
  poly: unknown;
  color: unknown;
  seen: number;
};

export function createFine(onChange: () => void, onError: (m: string) => void) {
  const size = Math.max(2, Math.min(6, (navigator.hardwareConcurrency || 4) - 2));
  const workers: { w: Worker; busy: number }[] = [];
  const chunks = new Map<string, Chunk>();
  const failed = new Set<string>();
  const flying = new Map<number, string>();
  /** Per resolution: every cell's volume summed over all the chunks in hand, and its pixels. */
  const totalsBy = new Map<number, Map<bigint, { s: number; n: number }>>();
  const totalsOf = (res: number) => {
    let t = totalsBy.get(res);
    if (!t) totalsBy.set(res, (t = new Map()));
    return t;
  };
  let res = 18;
  let queue: string[] = [];
  let want: string[] = [];
  let active = false;
  let nextId = 1;
  let tick = 0;
  let ready = false;

  const key = (r: number, cy: number, cx: number) => `${r}/${cy}/${cx}`;
  const parts = (k: string) => k.split("/").map(Number) as [number, number, number];

  function recolor(k: string) {
    const c = chunks.get(k);
    if (!c || !c.own.length) return;
    const rgba = new Uint8Array(c.own.length * 4);
    const totals = totalsOf(parts(k)[0]);
    for (let i = 0; i < c.own.length; i++) {
      const t = totals.get(c.cells[c.own[i]!]!)!;
      // the fuel on the ramp's scale, in the red byte: the shader makes the color (geo.ts, risk.ts)
      rgba[4 * i] = Math.round(255 * volT(t.s / t.n));
      rgba[4 * i + 3] = 255;
    }
    c.color = colorData(rgba);
  }

  function drop(k: string) {
    const c = chunks.get(k);
    if (!c) return;
    const totals = totalsOf(parts(k)[0]);
    for (let i = 0; i < c.cells.length; i++) {
      const t = totals.get(c.cells[i]!);
      if (!t) continue;
      t.s -= c.sum[i]!;
      t.n -= c.n[i]!;
      if (t.n <= 0) totals.delete(c.cells[i]!);
    }
    chunks.delete(k);
  }

  function arrived(k: string, d: Extract<FineAnswer, { type: "chunk" }>) {
    const [r, cy, cx] = parts(k);
    const totals = totalsOf(r);
    for (let i = 0; i < d.cells.length; i++) {
      const t = totals.get(d.cells[i]!);
      if (t) {
        t.s += d.sum[i]!;
        t.n += d.n[i]!;
      } else totals.set(d.cells[i]!, { s: d.sum[i]!, n: d.n[i]! });
    }
    chunks.set(k, { cells: d.cells, sum: d.sum, n: d.n, own: d.own, tris: d.tris, ...polygonBatch(d.coords, d.rings), color: null, seen: tick });
    // the cells on its edges now have their other part: this chunk and the eight around it take their colors again
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) recolor(key(r, cy + dy, cx + dx));
    if (chunks.size > MAX_CHUNKS) {
      const keep = new Set(want);
      const old = [...chunks].filter(([q]) => !keep.has(q)).sort((a, b) => a[1].seen - b[1].seen);
      for (const [q] of old) {
        if (chunks.size <= MAX_CHUNKS) break;
        drop(q);
      }
    }
  }

  function pump() {
    if (!ready) return;
    while (queue.length) {
      const w = workers.reduce((a, b) => (b.busy < a.busy ? b : a));
      if (w.busy >= LANES) return;
      const k = queue.shift()!;
      if (chunks.has(k) || failed.has(k) || [...flying.values()].includes(k)) continue;
      const [r, cy, cx] = parts(k);
      const id = nextId++;
      flying.set(id, k);
      w.busy++;
      w.w.postMessage({ type: "read", id, cy, cx, res: r } satisfies FineAsk);
    }
  }

  (async () => {
    const head = (await (await fetch(new URL("models/fuel_volume.json", new URL("./", location.href)))).json()) as Head;
    let up = 0;
    for (let i = 0; i < size; i++) {
      const ent = { w: new FineWorker(), busy: 0 };
      ent.w.onmessage = (ev: MessageEvent<FineAnswer>) => {
        const d = ev.data;
        if (d.type === "ready") {
          if (d.probeGap > 0.01) onError(`volume head differs from its export by ${(100 * d.probeGap).toFixed(1)} %`);
          if (++up === size) {
            ready = true;
            pump();
          }
          return;
        }
        const k = flying.get(d.id);
        flying.delete(d.id);
        ent.busy--;
        if (k != null) {
          if (d.type === "chunk") arrived(k, d);
          else {
            failed.add(k);
            onError(`embeddings: ${d.message}`);
          }
        }
        pump();
        onChange();
      };
      ent.w.postMessage({ type: "init", head, root: ROOT, t: YEAR - 2017 } satisfies FineAsk);
      workers.push(ent);
    }
  })().catch((e) => onError(`volume head: ${e?.message ?? e}`));

  return {
    /** Ask for the fine cells of a view (west, south, east, north); below FINE_ZOOM nothing is read or drawn. */
    view(zoom: number, bbox: [number, number, number, number]) {
      active = zoom >= FINE_ZOOM;
      if (!active) {
        want = [];
        queue = [];
        return;
      }
      tick++;
      res = fineRes(zoom);
      const cx0 = Math.floor((bbox[0] - X0) / RES / CHUNK), cx1 = Math.floor((bbox[2] - X0) / RES / CHUNK);
      const cy0 = Math.floor((Y0 - bbox[3]) / RES / CHUNK), cy1 = Math.floor((Y0 - bbox[1]) / RES / CHUNK);
      const mx = (cx0 + cx1) / 2, my = (cy0 + cy1) / 2;
      const list: [number, string][] = [];
      for (let cy = cy0; cy <= cy1; cy++) for (let cx = cx0; cx <= cx1; cx++) list.push([(cx - mx) ** 2 + (cy - my) ** 2, key(res, cy, cx)]);
      // from the middle of the view outward
      want = list.sort((a, b) => a[0] - b[0]).map((l) => l[1]);
      for (const k of want) {
        const c = chunks.get(k);
        if (c) c.seen = tick;
      }
      queue = want.filter((k) => !chunks.has(k) && !failed.has(k));
      pump();
    },
    get active() {
      return active;
    },
    get res() {
      return res;
    },
    progress() {
      const got = want.filter((k) => chunks.has(k) || failed.has(k)).length;
      return { got, total: want.length, done: !active || got === want.length };
    },
    /** The fine cell under a point: its mean volume (m) and its 10 m pixels. */
    at(lon: number, lat: number) {
      if (!active) return null;
      const t = totalsOf(res).get(lonLatToCell([lon, lat] as never, res));
      return t ? { vol: t.s / t.n, n: t.n, res } : null;
    },
    layers(beforeId: string | undefined) {
      if (!active) return [];
      const out = [];
      for (const k of want) {
        // until a chunk is in at this resolution, the same chunk at the other one stays up if it is in hand
        const [r, cy, cx] = parts(k);
        const other = key(r === 19 ? 18 : 19, cy, cx);
        const id = chunks.get(k)?.color ? k : other;
        const c = chunks.get(id);
        if (c?.color && c.own.length) out.push(cellLayer(`fine-${id}`, c as Chunk & { color: unknown }, beforeId));
      }
      return out;
    },
  };
}
