import { PathLayer } from "@deck.gl/layers";

import GoesWorker from "./goes.worker?worker";
import { NX, NY, type GoesAnswer, type GoesAsk } from "./goes.worker";

/**
 * The fires GOES-West saw, on the map: one white shape per fire.
 *
 * The scans of the weather's window are listed from NOAA's bucket and read in workers (goes.worker.ts), one scan
 * every EVERY minutes, oldest first, so fires arrive in the order they happened. A fire pixel is the satellite's own
 * grid cell, about 2 km wide and 3 km tall over California; the fire is somewhere inside it.
 *
 * A fire is drawn as the outer edge of its pixels and nothing else: an edge two pixels share is dropped, so a fire of
 * many pixels is one outline, not a grid of boxes, and the ground under it stays in view. Two outlines at the time
 * shown:
 *   bright  around the pixels seen burning within the last HOT hours
 *   faint   around every pixel seen so far in the window
 * The color is a flame's blue: the risk ramp already runs through the oranges and yellows, and blue against orange
 * holds for eyes that are weak in red.
 */

const LIST = "https://noaa-goes18.s3.amazonaws.com/?list-type=2&prefix=ABI-L2-FDCC/";
const EVERY = 15;
const WORKERS = 3;
const HOT = 3 * 3_600_000;
const FLAME: [number, number, number] = [70, 190, 255];
type Edge = { path: [number, number][] };

// GOES-West's fixed grid for the lower 48 (GOES-R Product User Guide): scan angles of pixel centers, radians
const X0 = -0.069972, DX = 0.000056, Y0 = 0.128212, DY = -0.000056;
const REQ = 6378137, RPOL = 6356752.31414, H = 35786023 + REQ, LON0 = (-137 * Math.PI) / 180;

/** Scan angles (x east, y north) to lon / lat in degrees; NaN off the earth. */
function lonLat(x: number, y: number): [number, number] {
  const cx = Math.cos(x), sx = Math.sin(x), cy = Math.cos(y), sy = Math.sin(y);
  const a = sx * sx + cx * cx * (cy * cy + ((REQ * REQ) / (RPOL * RPOL)) * sy * sy);
  const b = -2 * H * cx * cy;
  const rs = (-b - Math.sqrt(b * b - 4 * a * (H * H - REQ * REQ))) / (2 * a);
  const px = rs * cx * cy, py = -rs * sx, pz = rs * cx * sy;
  return [((LON0 - Math.atan(py / (H - px))) * 180) / Math.PI, (Math.atan(((REQ * REQ) / (RPOL * RPOL)) * pz / Math.hypot(H - px, py)) * 180) / Math.PI];
}

/** The grid pixel (row, col) holding a place, or null off the grid. */
function pixelAt(lon: number, lat: number): number | null {
  const la = (lat * Math.PI) / 180, lo = (lon * Math.PI) / 180;
  const latc = Math.atan(((RPOL * RPOL) / (REQ * REQ)) * Math.tan(la));
  const rc = RPOL / Math.sqrt(1 - (1 - (RPOL * RPOL) / (REQ * REQ)) * Math.cos(latc) ** 2);
  const sx = H - rc * Math.cos(latc) * Math.cos(lo - LON0), sy = -rc * Math.cos(latc) * Math.sin(lo - LON0), sz = rc * Math.sin(latc);
  const x = Math.asin(-sy / Math.hypot(sx, sy, sz)), y = Math.atan(sz / sx);
  const col = Math.round((x - X0) / DX), row = Math.round((y - Y0) / DY);
  return col >= 0 && col < NX && row >= 0 && row < NY ? row * NX + col : null;
}

/** When a scan started (ms), from its file name (..._sYYYYDDDHHMMSSt_e...). */
function scanTime(key: string): number {
  const s = key.split("_s")[1]!;
  return Date.UTC(Number(s.slice(0, 4)), 0, Number(s.slice(4, 7)), Number(s.slice(7, 9)), Number(s.slice(9, 11)), Number(s.slice(11, 13)));
}

export type FirePixel = { id: number; poly: [number, number][]; times: number[] };
export type FireAt = { first: number; last: number; scans: number; hot: boolean };

export function createGoes(onChange: () => void, onError: (m: string) => void) {
  let workers: Worker[] = [];
  let px = new Map<number, FirePixel>();
  let list: FirePixel[] = [];
  let span: [number, number] | null = null;
  let run = 0, got = 0, total = 0, failed = 0, listed = false;
  let on = true;

  /** A pixel at a time: 2 burning now, 1 seen earlier, 0 not yet. */
  const stateAt = (p: FirePixel, t: number) => {
    const ts = p.times;
    if (ts[0]! > t) return 0;
    let last = ts[0]!;
    for (let i = ts.length - 1; i > 0; i--) if (ts[i]! <= t) { last = ts[i]!; break; }
    return t - last <= HOT ? 2 : 1;
  };

  function add(t: number, ids: Int32Array) {
    for (const id of ids) {
      let p = px.get(id);
      if (!p) {
        const row = Math.floor(id / NX), col = id % NX;
        const corner = (dr: number, dc: number) => lonLat(X0 + DX * (col + dc), Y0 + DY * (row + dr));
        const poly = [corner(-0.5, -0.5), corner(-0.5, 0.5), corner(0.5, 0.5), corner(0.5, -0.5)];
        if (poly.some((c) => c[0] !== c[0] || c[1] !== c[1])) continue;
        px.set(id, (p = { id, poly, times: [] }));
      }
      // scans come from several workers, not in order
      let i = p.times.length;
      while (i > 0 && p.times[i - 1]! > t) i--;
      if (p.times[i - 1] !== t) p.times.splice(i, 0, t);
    }
  }

  async function listKeys(t0: number, t1: number): Promise<string[]> {
    const keys: string[] = [];
    const end = Math.min(t1, Date.now());
    for (let d = Date.UTC(new Date(t0).getUTCFullYear(), new Date(t0).getUTCMonth(), new Date(t0).getUTCDate()); d <= end; d += 86_400_000) {
      const day = new Date(d), doy = Math.round((d - Date.UTC(day.getUTCFullYear(), 0, 1)) / 86_400_000) + 1;
      const r = await fetch(`${LIST}${day.getUTCFullYear()}/${String(doy).padStart(3, "0")}/`);
      if (!r.ok) throw new Error(`listing GOES scans: HTTP ${r.status}`);
      for (const m of (await r.text()).matchAll(/<Key>([^<]+\.nc)<\/Key>/g)) {
        const k = m[1]!;
        // a scan written twice at a site handover carries start == end in its name
        if (k.split("_s")[1]!.slice(0, 13) !== k.split("_e")[1]!.slice(0, 13)) keys.push(k);
      }
    }
    // one scan per EVERY minutes: the first in each slot
    const seen = new Set<number>();
    return keys
      .map((k) => [scanTime(k), k] as const)
      .filter(([t]) => t >= t0 && t <= end)
      .sort((a, b) => a[0] - b[0])
      .filter(([t]) => {
        const slot = Math.floor(t / (EVERY * 60_000));
        return seen.has(slot) ? false : (seen.add(slot), true);
      })
      .map(([, k]) => k);
  }

  let pending = 0;
  const changed = () => {
    // many scans land close together: one redraw for the lot
    if (pending) return;
    pending = window.setTimeout(() => {
      pending = 0;
      list = [...px.values()];
      onChange();
    }, 250);
  };

  return {
    get on() {
      return on;
    },
    set on(v: boolean) {
      on = v;
      onChange();
    },
    /** Read the scans of a span of time (ms); the same span again does nothing. */
    window(t0: number, t1: number) {
      if (span && span[0] === t0 && span[1] === t1) return;
      span = [t0, t1];
      const mine = ++run;
      for (const w of workers) w.terminate();
      workers = [];
      px = new Map();
      list = [];
      got = total = failed = 0;
      listed = false;
      onChange();
      listKeys(t0, t1)
        .then((keys) => {
          if (mine !== run) return;
          listed = true;
          total = keys.length;
          for (let k = 0; k < WORKERS; k++) {
            const w = new GoesWorker();
            w.onmessage = (ev: MessageEvent<GoesAnswer>) => {
              const d = ev.data;
              if (d.id !== mine) return;
              got++;
              if (d.type === "scan") add(scanTime(d.key), d.px);
              else if (++failed === 1) onError(`GOES scan not read: ${d.message}`);
              changed();
            };
            w.postMessage({ type: "scans", id: mine, keys: keys.filter((_, i) => i % WORKERS === k) } satisfies GoesAsk);
            workers.push(w);
          }
          onChange();
        })
        .catch((e) => {
          if (mine !== run) return;
          listed = true;
          onError(e instanceof Error ? e.message : String(e));
          onChange();
        });
    },
    progress: () => ({ got, total, done: listed && got >= total, failed }),
    /** Fire pixels seen by the time t, and how many of them are burning then. */
    count(t: number) {
      let seen = 0, hot = 0;
      for (const p of list) {
        const s = stateAt(p, t);
        if (s) seen++;
        if (s === 2) hot++;
      }
      return { seen, hot };
    },
    /** The fire pixel over a place at time t, if GOES has seen one there by then. */
    at(lon: number, lat: number, t: number): FireAt | null {
      const id = pixelAt(lon, lat);
      const p = id == null ? undefined : px.get(id);
      if (!p || !on) return null;
      const s = stateAt(p, t);
      if (!s) return null;
      const upTo = p.times.filter((x) => x <= t);
      return { first: upTo[0]!, last: upTo[upTo.length - 1]!, scans: upTo.length, hot: s === 2 };
    },
    layers(t: number) {
      if (!on || !list.length) return [];
      // each pixel's four edges, keyed by the grid corners they join; an edge met twice is inside the shape
      const outline = (least: number): Edge[] => {
        const edges = new Map<number, [number, number][] | null>();
        for (const p of list) {
          if (stateAt(p, t) < least) continue;
          const row = Math.floor(p.id / NX), col = p.id % NX;
          const c = [row * (NX + 1) + col, row * (NX + 1) + col + 1, (row + 1) * (NX + 1) + col + 1, (row + 1) * (NX + 1) + col];
          for (let k = 0; k < 4; k++) {
            const a = c[k]!, b = c[(k + 1) % 4]!;
            const key = Math.min(a, b) * 2 + (Math.abs(a - b) === 1 ? 0 : 1);
            edges.set(key, edges.has(key) ? null : [p.poly[k]!, p.poly[(k + 1) % 4]!]);
          }
        }
        const out: Edge[] = [];
        for (const e of edges.values()) if (e) out.push({ path: e });
        return out;
      };
      const line = (id: string, data: Edge[], width: number, alpha: number) =>
        new PathLayer<Edge>({ id, data, getPath: (d) => d.path, getColor: [...FLAME, alpha], getWidth: width, widthUnits: "pixels", capRounded: true, jointRounded: true, pickable: false });
      return [line("goes-seen", outline(1), 1.25, 120), line("goes-burning", outline(2), 2.75, 255)];
    },
  };
}
