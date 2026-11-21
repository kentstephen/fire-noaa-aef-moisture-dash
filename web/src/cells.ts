import { cellToChildren, lonLatToCell } from "a5-js";
import type { RecordBatch } from "apache-arrow";

import type { Answer, Ask } from "./cells.worker";
import CellsWorker from "./cells.worker.ts?worker";
import { cellLayer, colorData, polygonBatch } from "./geo";

/**
 * The fuel cells on the map. One Parquet file per A5 resolution (scripts/levels.py), sorted by cell ID; the view
 * picks the resolution whose cells are about CELL_PX screen pixels across and reads only the row groups that hold
 * cells in view. Each row group is one GeoArrow polygon batch drawn by GeoArrowSolidPolygonLayer.
 */

/** Resolutions baked. */
export const RES_LO = 8;
export const RES_HI = 15;
/** A cell is at least this many screen pixels across: at 1 a full window is about 1.4 M cells, the most read for one view. */
const CELL_PX = 1;
/** Row groups kept in hand (about 3.5 MB each) before the ones seen longest ago are let go. */
const MAX_GROUPS = 140;
const EARTH_M2 = 5.1007e14;
/** A cell's width in m: the side of a square of its area. */
export const cellM = (res: number) => Math.sqrt(EARTH_M2 / (60 * 4 ** (res - 1)));

/**
 * The resolution for a zoom. It goes by the zoom alone (the screen scale at California's middle), so panning north or
 * south never changes it, and it holds the resolution on screen until the zoom is clearly past the step (a fifth of
 * a zoom level either way), so a small nudge of the zoom does not flip it back and forth.
 */
const REF_LAT = 37;
const HOLD = 0.2;
function plainRes(zoom: number) {
  const mpp = (78271.517 * Math.cos((REF_LAT * Math.PI) / 180)) / 2 ** zoom;
  for (let r = RES_HI; r > RES_LO; r--) if (cellM(r) >= CELL_PX * mpp) return r;
  return RES_LO;
}
export function resFor(zoom: number, now = 0) {
  const lo = plainRes(zoom - HOLD), hi = plainRes(zoom + HOLD);
  return now >= lo && now <= hi ? now : plainRes(zoom);
}
/** A coarse level kept whole and drawn under everything, so ground not read yet is coarse, never empty. */
const BASE_RES = 11;

type Group = {
  res: number;
  rg: number;
  cells: BigUint64Array;
  vol: Float32Array;
  n: Int32Array;
  batch: RecordBatch;
  poly: unknown;
  color: unknown;
  tris: Uint32Array;
  seen: number;
};
type Want = { res: number; rgs: number[] };

export function createCells(onChange: () => void, onError: (m: string) => void) {
  const worker = new CellsWorker();
  const ask = (a: Ask) => worker.postMessage(a);
  ask({ type: "init", base: new URL("./", location.href).href });

  /** Per resolution: each row group's first and last cell ID, in order. */
  const index = new Map<number, { lo: BigUint64Array; hi: BigUint64Array }>();
  const groups = new Map<string, Group>();
  const failed = new Set<string>();
  let want: Want = { res: 0, rgs: [] };
  /** The last view that was whole, drawn underneath until the new one is. */
  let back: Want = { res: 0, rgs: [] };
  let pendingView: { zoom: number; bbox: [number, number, number, number] } | null = null;
  let tick = 0;

  const key = (res: number, rg: number) => `${res}/${rg}`;
  const whole = (w: Want) => w.res > 0 && w.rgs.every((rg) => groups.has(key(w.res, rg)) || failed.has(key(w.res, rg)));

  worker.onmessage = (ev: MessageEvent<Answer>) => {
    const d = ev.data;
    if (d.type === "index") {
      for (let r = RES_LO; r <= RES_HI; r++) {
        const at: number[] = [];
        for (let i = 0; i < d.res.length; i++) if (d.res[i] === r) at.push(i);
        index.set(r, { lo: BigUint64Array.from(at, (i) => d.lo[i]!), hi: BigUint64Array.from(at, (i) => d.hi[i]!) });
      }
      if (pendingView) view(pendingView.zoom, pendingView.bbox);
      else ask({ type: "want", jobs: base().map((rg) => ({ res: BASE_RES, rg })) });
    } else if (d.type === "group") {
      groups.set(key(d.res, d.rg), { res: d.res, rg: d.rg, cells: d.cells, vol: d.vol, n: d.n, tris: d.tris, ...polygonBatch(d.coords, d.rings), color: colorData(d.colors), seen: tick });
      trim();
      onChange();
    } else {
      if (d.res != null && d.rg != null) failed.add(key(d.res, d.rg));
      onError(d.message);
      onChange();
    }
  };

  const base = () => [...(index.get(BASE_RES)?.lo.keys() ?? [])];

  function trim() {
    if (groups.size <= MAX_GROUPS) return;
    const keep = new Set([...want.rgs.map((rg) => key(want.res, rg)), ...back.rgs.map((rg) => key(back.res, rg)), ...base().map((rg) => key(BASE_RES, rg))]);
    const old = [...groups].filter(([k]) => !keep.has(k)).sort((a, b) => a[1].seen - b[1].seen);
    for (const [k] of old) {
      if (groups.size <= MAX_GROUPS) break;
      groups.delete(k);
    }
  }

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

  /** The row groups of a resolution that hold cells inside a box. */
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

  function view(zoom: number, bbox: [number, number, number, number]) {
    pendingView = { zoom, bbox };
    if (!index.size) return;
    tick++;
    const res = resFor(zoom, want.res);
    const rgs = rowGroups(res, bbox);
    if (res !== want.res && want.res && (whole(want) || !back.res)) back = want;
    want = { res, rgs };
    for (const rg of rgs) {
      const g = groups.get(key(res, rg));
      if (g) g.seen = tick;
    }
    if (whole(want)) back = { res: 0, rgs: [] };
    const todo = (r: number, list: number[]) => list.filter((rg) => !groups.has(key(r, rg)) && !failed.has(key(r, rg))).map((rg) => ({ res: r, rg }));
    ask({ type: "want", jobs: [...todo(BASE_RES, base()), ...todo(res, rgs)] });
    onChange();
  }

  return {
    /** Ask for the cells of a view (west, south, east, north). */
    view,
    get res() {
      return want.res;
    },
    progress() {
      const got = want.rgs.filter((rg) => groups.has(key(want.res, rg))).length;
      return { got, total: want.rgs.length, ready: index.size > 0, done: whole(want) };
    },
    /** The cell under a point at the resolution on screen: its mean volume (m) and its 10 m pixels. */
    at(lon: number, lat: number) {
      const ix = index.get(want.res);
      if (!ix) return null;
      const c = lonLatToCell([lon, lat] as never, want.res);
      for (const rg of want.rgs) {
        if (ix.lo[rg]! > c || ix.hi[rg]! < c) continue;
        const g = groups.get(key(want.res, rg));
        if (!g) return null;
        let a = 0, b = g.cells.length - 1;
        while (a <= b) {
          const m = (a + b) >> 1;
          if (g.cells[m]! === c) return { vol: g.vol[m]!, n: g.n[m]!, res: want.res };
          if (g.cells[m]! < c) a = m + 1;
          else b = m - 1;
        }
      }
      return null;
    },
    /** The layers to draw: the last whole view underneath while the new one is still coming in, then the new one. */
    layers(beforeId: string | undefined) {
      if (whole(want) && back.res) back = { res: 0, rgs: [] };
      const out = [];
      const under: Want[] = want.res > BASE_RES ? [{ res: BASE_RES, rgs: base() }] : [];
      if (back.res && back.res !== want.res && back.res !== BASE_RES) under.push(back);
      for (const w of [...under, want]) {
        for (const rg of w.rgs) {
          const g = groups.get(key(w.res, rg));
          if (!g) continue;
          out.push(cellLayer(`fuel-${w.res}-${rg}`, g, beforeId));
        }
      }
      return out;
    },
  };
}
