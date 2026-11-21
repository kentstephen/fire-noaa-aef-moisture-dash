import { BLOCK, blockRowCol, HRRR } from "./wx.grid";
import type { WxAnswer, WxAsk } from "./wx.worker";
import WxWorker from "./wx.worker.ts?worker";

/**
 * The weather's say over the fuel map: whether the fuel in a place is primed. It is primed when Fosberg, built up
 * over days (wx.worker.ts), is at or over a line; the map shows primed cells in the ramp's color and the rest in
 * gray. Read from the HRRR analysis in the browser, only the blocks the cells on screen fall in.
 */

/** The moisture lags offered (hours), and the blocks of pixels a side. */
export const LAGS = [10, 100];
const PER = BLOCK * BLOCK;
const WORKERS = 3;
/** Blocks in flight per worker (five variables each). */
const LANES = 2;

export function createWeather(onChange: () => void, onError: (m: string) => void) {
  const workers: { w: Worker; busy: number; ready: boolean }[] = [];
  const blocks = new Map<number, Float32Array>();
  const failed = new Set<number>();
  const flying = new Map<number, number>();
  let days: number[] = [];
  let queue: number[] = [];
  let want = new Set<number>();
  let nextId = 1;
  /** bumps whenever what value() would answer may have changed */
  let rev = 0;
  const st = { on: true, day: -1, lag: 1, line: 20 };
  /** The index on the whole grid for the day and lag shown, put together from the blocks in hand; NaN elsewhere. */
  const field = new Float32Array(HRRR.w * HRRR.h).fill(NaN);
  let fieldRev = -1;
  function fill() {
    if (fieldRev === rev) return;
    fieldRev = rev;
    field.fill(NaN);
    if (st.day < 0) return;
    for (const [b, f] of blocks) {
      const [by, bx] = blockRowCol(b);
      const at = (st.day * LAGS.length + st.lag) * PER;
      const h = Math.min(BLOCK, HRRR.h - by * BLOCK), w = Math.min(BLOCK, HRRR.w - bx * BLOCK);
      for (let r = 0; r < h; r++) field.set(f.subarray(at + r * BLOCK, at + r * BLOCK + w), (by * BLOCK + r) * HRRR.w + bx * BLOCK);
    }
  }

  function pump() {
    while (queue.length) {
      const w = workers.filter((x) => x.ready).reduce<(typeof workers)[number] | null>((a, b) => (!a || b.busy < a.busy ? b : a), null);
      if (!w || w.busy >= LANES) return;
      const b = queue.shift()!;
      if (blocks.has(b) || failed.has(b) || [...flying.values()].includes(b)) continue;
      const id = nextId++;
      flying.set(id, b);
      w.busy++;
      w.w.postMessage({ type: "block", id, block: b } satisfies WxAsk);
    }
  }

  function start() {
    if (workers.length) return;
    for (let i = 0; i < WORKERS; i++) {
      const ent = { w: new WxWorker(), busy: 0, ready: false };
      ent.w.onmessage = (ev: MessageEvent<WxAnswer>) => {
        const d = ev.data;
        if (d.type === "ready") {
          ent.ready = true;
          if (!days.length) {
            days = d.days;
            if (st.day < 0) st.day = days.length - 1;
          }
        } else {
          if (d.id != null) {
            const b = flying.get(d.id);
            flying.delete(d.id);
            ent.busy--;
            if (d.type === "block") blocks.set(d.block, d.fos);
            else if (b != null) failed.add(b);
          }
          if (d.type === "error") onError(`weather: ${d.message}`);
        }
        rev++;
        pump();
        onChange();
      };
      ent.w.onerror = (e) => onError(`weather worker: ${e.message || "failed to start"} ${e.filename ? `(${e.filename.split("/").pop()}:${e.lineno})` : ""}`);
      ent.w.postMessage({ type: "init" } satisfies WxAsk);
      workers.push(ent);
    }
  }

  return {
    st,
    get days() {
      return days;
    },
    get rev() {
      return rev;
    },
    /** Change a setting (on, day, lag, line) and repaint. */
    set(patch: Partial<typeof st>) {
      Object.assign(st, patch);
      if (days.length) st.day = Math.min(days.length - 1, Math.max(0, st.day));
      rev++;
      if (st.on) start();
      onChange();
    },
    /** The blocks the cells on screen fall in, nearest the middle first. */
    need(list: number[]) {
      want = new Set(list);
      if (!st.on) return;
      start();
      queue = list.filter((b) => !blocks.has(b) && !failed.has(b));
      pump();
    },
    progress() {
      let got = 0;
      for (const b of want) if (blocks.has(b) || failed.has(b)) got++;
      return { got, total: want.size, done: !st.on || (days.length > 0 && got === want.size) };
    },
    /**
     * Fosberg, built up, at a place on the weather grid (column, row, fractions of a pixel) on the day and lag
     * shown, blended between the four pixels around it; NaN where the weather is not in hand.
     */
    value(col: number, row: number): number {
      if (col !== col) return NaN;
      fill();
      const c0 = Math.min(HRRR.w - 2, Math.floor(col)), r0 = Math.min(HRRR.h - 2, Math.floor(row));
      const fc = col - c0, fr = row - r0, i = r0 * HRRR.w + c0;
      const a = field[i]!, b = field[i + 1]!, c = field[i + HRRR.w]!, d = field[i + HRRR.w + 1]!;
      if (a !== a || b !== b || c !== c || d !== d) return field[Math.round(row) * HRRR.w + Math.round(col)]!;
      return (a * (1 - fc) + b * fc) * (1 - fr) + (c * (1 - fc) + d * fc) * fr;
    },
    /**
     * How far the fuel there is primed, 0 to 1: 0 under the line, 1 a point and a half over it, so the edge
     * between gray and color is a short blend and not the outline of a weather pixel. 0 where not known yet.
     */
    lit(col: number, row: number): number {
      const v = this.value(col, row);
      return v === v ? Math.min(1, Math.max(0, (v - st.line) / 1.5)) : 0;
    },
  };
}
export type Weather = ReturnType<typeof createWeather>;
