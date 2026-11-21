// @ts-expect-error jsfive ships no types
import * as hdf5 from "jsfive";

/**
 * GOES-West fire scans read here, straight from NOAA's bucket (s3://noaa-goes18, product ABI-L2-FDCC: Fire / Hot
 * Spot Characterization, the lower 48, a scan every 5 minutes). A file is about 0.27 MB, so each is fetched whole
 * and opened as HDF5 in memory; nothing is indexed ahead of time. How a file stores its Mask (250 x 250 tiles, or
 * strips across the whole width, as from July to November 2026) is read from the file itself.
 *
 * Kept from a scan: the pixels whose Mask is a fire category, low probability left out.
 *   10 processed fire, 11 saturated, 12 cloud contaminated, 13 high, 14 medium probability (15 low: left out);
 *   30 to 34 the same, temporally filtered (35 low: left out).
 */

export const NX = 2500, NY = 1500;
const BASE = "https://noaa-goes18.s3.amazonaws.com/";
const FIRE = new Set([10, 11, 12, 13, 14, 30, 31, 32, 33, 34]);
/** Files fetched ahead of the one being opened. */
const AHEAD = 4;

export type GoesAsk = { type: "scans"; id: number; keys: string[] };
export type GoesAnswer =
  /** px: row * NX + col of each fire pixel in the scan */
  | { type: "scan"; id: number; key: string; px: Int32Array }
  | { type: "error"; id: number; key: string; message: string };

const post = (a: GoesAnswer, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(a, transfer);

async function get(key: string): Promise<ArrayBuffer> {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(BASE + key);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.arrayBuffer();
    } catch (e) {
      if (attempt === 3) throw e;
      await new Promise((ok) => setTimeout(ok, 1500 * (attempt + 1)));
    }
  }
}

function firePixels(buf: ArrayBuffer): Int32Array {
  const f = new hdf5.File(buf);
  const m = f.get("Mask");
  const shape = m.shape as number[];
  if (shape[0] !== NY || shape[1] !== NX) throw new Error(`Mask is ${shape.join(" x ")}, not ${NY} x ${NX}: a different grid`);
  const v = m.value as ArrayLike<number>;
  const out: number[] = [];
  for (let i = 0; i < v.length; i++) if (FIRE.has(v[i]!)) out.push(i);
  return Int32Array.from(out);
}

self.onmessage = async (ev: MessageEvent<GoesAsk>) => {
  const { id, keys } = ev.data;
  const flying = new Map<number, Promise<ArrayBuffer>>();
  const ask = (i: number) => {
    if (i < keys.length && !flying.has(i)) {
      const p = get(keys[i]!);
      p.catch(() => {});
      flying.set(i, p);
    }
  };
  for (let i = 0; i < keys.length; i++) {
    for (let k = i; k <= i + AHEAD; k++) ask(k);
    try {
      const px = firePixels(await flying.get(i)!);
      post({ type: "scan", id, key: keys[i]!, px }, [px.buffer]);
    } catch (e) {
      post({ type: "error", id, key: keys[i]!, message: e instanceof Error ? e.message : String(e) });
    }
    flying.delete(i);
  }
};
