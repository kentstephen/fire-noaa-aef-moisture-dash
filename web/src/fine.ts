import { lonLatToCell } from "a5-js";
import type { RecordBatch } from "apache-arrow";

import type { FineAnswer, FineAsk, Head } from "./fine.worker";
import FineWorker from "./fine.worker.ts?worker";
import { cellLayer, colorData, polygonBatch } from "./geo";
import { rampInto, volT } from "./ramp";

/**
 * Zoomed in past the baked cells: the real 10 m embeddings of the view are read and scored in the browser
 * (fine.worker.ts) and drawn as finer A5 cells over the baked ones, on the same ramp. Nothing is kept on disk.
 */

/** From this zoom the embeddings are read: a full window is about 50 to 120 chunks of 2.6 MB. */
export const FINE_ZOOM = 12.5;
/** The cells made: about 22 m across, six 10 m pixels each. One finer and some cells would hold no pixel. */
export const FINE_RES = 18;
/** AlphaEarth on Source Cooperative, through its gateway; the year's place on the time axis (2017 is 0). */
const ROOT = "https://data.source.coop/tge-labs/aef-mosaic";
const YEAR = 2025;
const RES = 8.983111749910169e-5, X0 = -180, Y0 = 83.68570533713473, CHUNK = 256;
/** Chunks kept (about 1.5 MB each as cells) before the ones seen longest ago are let go. */
const MAX_CHUNKS = 260;
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
  /** recolored this many times, so its layer takes the new colors */
  rev: number;
};

export function createFine(onChange: () => void, onError: (m: string) => void) {
  const size = Math.max(2, Math.min(6, (navigator.hardwareConcurrency || 4) - 2));
  const workers: { w: Worker; busy: number }[] = [];
  const chunks = new Map<string, Chunk>();
  const failed = new Set<string>();
  const flying = new Map<number, string>();
  /** Every cell's volume summed over all the chunks in hand, and its pixels. */
  const totals = new Map<bigint, { s: number; n: number }>();
  let queue: string[] = [];
  let want: string[] = [];
  let active = false;
  let nextId = 1;
  let tick = 0;
  let ready = false;

  const key = (cy: number, cx: number) => `${cy}/${cx}`;

  function recolor(k: string) {
    const c = chunks.get(k);
    if (!c || !c.own.length) return;
    const rgba = new Uint8Array(c.own.length * 4);
    for (let i = 0; i < c.own.length; i++) {
      const t = totals.get(c.cells[c.own[i]!]!)!;
      rampInto(rgba, 4 * i, volT(t.s / t.n));
    }
    c.color = colorData(rgba);
    c.rev++;
  }

  function drop(k: string) {
    const c = chunks.get(k);
    if (!c) return;
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
    for (let i = 0; i < d.cells.length; i++) {
      const t = totals.get(d.cells[i]!);
      if (t) {
        t.s += d.sum[i]!;
        t.n += d.n[i]!;
      } else totals.set(d.cells[i]!, { s: d.sum[i]!, n: d.n[i]! });
    }
    chunks.set(k, { cells: d.cells, sum: d.sum, n: d.n, own: d.own, tris: d.tris, ...polygonBatch(d.coords, d.rings), color: null, seen: tick, rev: 0 });
    // the cells on its edges now have their other part: this chunk and the eight around it take their colors again
    const [cy, cx] = k.split("/").map(Number) as [number, number];
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) recolor(key(cy + dy, cx + dx));
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
      const [cy, cx] = k.split("/").map(Number) as [number, number];
      const id = nextId++;
      flying.set(id, k);
      w.busy++;
      w.w.postMessage({ type: "read", id, cy, cx, res: FINE_RES } satisfies FineAsk);
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
      const cx0 = Math.floor((bbox[0] - X0) / RES / CHUNK), cx1 = Math.floor((bbox[2] - X0) / RES / CHUNK);
      const cy0 = Math.floor((Y0 - bbox[3]) / RES / CHUNK), cy1 = Math.floor((Y0 - bbox[1]) / RES / CHUNK);
      const mx = (cx0 + cx1) / 2, my = (cy0 + cy1) / 2;
      const list: [number, string][] = [];
      for (let cy = cy0; cy <= cy1; cy++) for (let cx = cx0; cx <= cx1; cx++) list.push([(cx - mx) ** 2 + (cy - my) ** 2, key(cy, cx)]);
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
    progress() {
      const got = want.filter((k) => chunks.has(k) || failed.has(k)).length;
      return { got, total: want.length, done: !active || got === want.length };
    },
    /** The fine cell under a point: its mean volume (m) and its 10 m pixels. */
    at(lon: number, lat: number) {
      if (!active) return null;
      const t = totals.get(lonLatToCell([lon, lat] as never, FINE_RES));
      return t ? { vol: t.s / t.n, n: t.n, res: FINE_RES } : null;
    },
    layers(beforeId: string | undefined) {
      if (!active) return [];
      const out = [];
      for (const k of want) {
        const c = chunks.get(k);
        if (c?.color && c.own.length) out.push(cellLayer(`fine-${k}`, c as Chunk & { color: unknown }, beforeId));
      }
      return out;
    },
  };
}
