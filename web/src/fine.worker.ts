import { cellToBoundary, cellToLonLat, lonLatToCell } from "a5-js";
import { Zstd } from "numcodecs";

import { fan } from "./geo.fan";
import { hrrrXY } from "./wx.grid";

/**
 * Cells finer than the baked ones, made here from the 10 m AlphaEarth embeddings: one 256 px chunk of the mosaic is
 * read (a range of its shard, as aef-biomass zread.ts reads it), the volume head runs on every pixel (the same
 * arithmetic as scripts/bake_fuel.py), and each pixel's volume goes to the A5 cell its center falls in.
 *
 * What goes back: every cell the chunk's pixels touch with its sum and count (a cell on the chunk's edge gets the
 * rest from the neighbor chunk, added up on the page), and the pentagons of the cells whose centers are inside the
 * chunk, which are the ones this chunk draws.
 */

export type Head = { n_in: number; hidden: number; n_out: number; vol_lo: number; vol_hi: number; mean: number[]; scale: number[]; w1: number[]; b1: number[]; w2: number[]; b2: number[]; probe_q: number[]; probe_out: number[] };
export type FineAsk = { type: "init"; head: Head; root: string; t: number } | { type: "read"; id: number; cy: number; cx: number; res: number };
export type FineAnswer =
  | { type: "ready"; probeGap: number }
  | { type: "chunk"; id: number; cells: BigUint64Array; sum: Float32Array; n: Uint16Array; own: Int32Array; wx: Float32Array; coords: Float64Array; rings: Int32Array; tris: Uint32Array }
  | { type: "error"; id: number; message: string };

/** The mosaic's grid (EPSG:4326) and its chunks (aef-biomass aef.ts). */
const RES = 8.983111749910169e-5, X0 = -180, Y0 = 83.68570533713473;
const SHARD = 4096, CHUNK = 256, PER = SHARD / CHUNK, N = CHUNK * CHUNK, BANDS = 64;
const INDEX_BYTES = PER * PER * 16 + 4;
const EMPTY = 0xffffffffffffffffn;

const post = (a: FineAnswer, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(a, transfer);
const zstd = Zstd.fromConfig({ id: "zstd", level: 3 });

let root = "", t = 0;
let H: Head;
/** Each band's 256 possible bytes already dequantized (sign(q) (q / 127.5)^2) and standardized. */
let lut: Float32Array;
let W1: Float32Array, B1: Float32Array, W2vol: Float32Array, b2vol = 0;

const fetchRange = async (url: string, range: string) => {
  const r = await fetch(url, { headers: { Range: `bytes=${range}` }, cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} for ${url}`);
  return new Uint8Array(await r.arrayBuffer());
};
const shardIndex = new Map<string, Promise<DataView>>();
function index(url: string) {
  let p = shardIndex.get(url);
  if (!p) {
    p = fetchRange(url, `-${INDEX_BYTES}`).then((b) => {
      if (b.byteLength !== INDEX_BYTES) throw new Error(`shard index of ${b.byteLength} bytes`);
      return new DataView(b.buffer, b.byteOffset, b.byteLength);
    });
    shardIndex.set(url, p);
    p.catch(() => shardIndex.delete(url));
  }
  return p;
}

/** A chunk's embeddings, band-major int8 [64, 65536]; null where the mosaic has none. */
async function embeddings(cy: number, cx: number): Promise<Int8Array | null> {
  const url = `${root}/embeddings/c/${t}/0/${Math.floor(cy / PER)}/${Math.floor(cx / PER)}`;
  const at = 16 * ((cy % PER) * PER + (cx % PER));
  const view = await index(url);
  const off = view.getBigUint64(at, true), len = view.getBigUint64(at + 8, true);
  if (off === EMPTY && len === EMPTY) return null;
  const d = await zstd.decode(await fetchRange(url, `${off}-${off + len - 1n}`));
  if (d.byteLength !== BANDS * N) throw new Error(`chunk of ${d.byteLength} bytes`);
  return new Int8Array(d.buffer, d.byteOffset, d.byteLength);
}

/** Volume in m (cover x height) of n pixels of band-major int8, the pixels a stride apart; NaN where there is no embedding. */
function volume(e: Int8Array, n: number): Float32Array {
  const out = new Float32Array(n);
  const BLOCK = 4096, hid = H.hidden;
  const x = new Float32Array(BANDS * BLOCK), acc = new Float32Array(BLOCK), o = new Float32Array(BLOCK);
  const span = H.vol_hi - H.vol_lo;
  for (let s = 0; s < n; s += BLOCK) {
    const m = Math.min(BLOCK, n - s);
    for (let b = 0; b < BANDS; b++) {
      const row = b * n + s, l = b * 256 + 128, xo = b * BLOCK;
      for (let i = 0; i < m; i++) x[xo + i] = lut[l + e[row + i]!]!;
    }
    o.fill(b2vol, 0, m);
    for (let j = 0; j < hid; j++) {
      acc.fill(B1[j]!, 0, m);
      for (let b = 0; b < BANDS; b++) {
        const w = W1[j * BANDS + b]!, xo = b * BLOCK;
        for (let i = 0; i < m; i++) acc[i]! += w * x[xo + i]!;
      }
      const w2 = W2vol[j]!;
      for (let i = 0; i < m; i++) if (acc[i]! > 0) o[i]! += w2 * acc[i]!;
    }
    for (let i = 0; i < m; i++) out[s + i] = e[s + i] === -128 ? NaN : Math.max(Math.exp(o[i]! * span + H.vol_lo) - 0.02, 0);
  }
  return out;
}

/** The head's first output (the scaled log volume) for the export's own probe pixels, against the export's answers. */
function probeGap(): number {
  const n = H.probe_q.length / BANDS;
  const e = new Int8Array(H.probe_q.length);
  for (let p = 0; p < n; p++) for (let b = 0; b < BANDS; b++) e[b * n + p] = H.probe_q[p * BANDS + b]!;
  const v = volume(e, n);
  const span = H.vol_hi - H.vol_lo;
  let worst = 0;
  for (let p = 0; p < n; p++) {
    if (e[p] === -128) continue;
    const want = Math.max(Math.exp(H.probe_out[p * H.n_out]! * span + H.vol_lo) - 0.02, 0);
    worst = Math.max(worst, Math.abs(v[p]! - want) / Math.max(want, 0.05));
  }
  return worst;
}

async function read(id: number, cy: number, cx: number, res: number) {
  const e = await embeddings(cy, cx);
  if (!e) {
    post({ type: "chunk", id, cells: new BigUint64Array(0), sum: new Float32Array(0), n: new Uint16Array(0), own: new Int32Array(0), wx: new Float32Array(0), coords: new Float64Array(0), rings: new Int32Array(1), tris: new Uint32Array(0) });
    return;
  }
  const vol = volume(e, N);
  const acc = new Map<bigint, { s: number; n: number }>();
  const pt: [number, number] = [0, 0];
  // a cell about the size of a pixel (resolution 19, 11 m) would sometimes hold no pixel center: there each pixel is
  // counted at four points inside it, so every cell is reached and a cell across two pixels gets both
  const sub = res >= 19 ? [0.25, 0.75] : [0.5];
  for (let y = 0; y < CHUNK; y++) {
    for (let x = 0; x < CHUNK; x++) {
      const v = vol[y * CHUNK + x]!;
      if (v !== v) continue;
      for (const dy of sub) {
        pt[1] = Y0 - (cy * CHUNK + y + dy) * RES;
        for (const dx of sub) {
          pt[0] = X0 + (cx * CHUNK + x + dx) * RES;
          const c = lonLatToCell(pt as never, res);
          const a = acc.get(c);
          if (a) {
            a.s += v;
            a.n++;
          } else acc.set(c, { s: v, n: 1 });
        }
      }
    }
  }
  const count = acc.size;
  const cells = BigUint64Array.from(acc.keys()).sort();
  const sum = new Float32Array(count), n = new Uint16Array(count);
  // the cells this chunk draws: the ones whose centers are inside it
  const w = X0 + cx * CHUNK * RES, east = w + CHUNK * RES, north = Y0 - cy * CHUNK * RES, south = north - CHUNK * RES;
  const own: number[] = [];
  let wx = new Float32Array(count * 2);
  let coords = new Float64Array(count * 12), tris = new Uint32Array(count * 9);
  const rings: number[] = [0];
  let p = 0, q = 0;
  for (let i = 0; i < count; i++) {
    const a = acc.get(cells[i]!)!;
    sum[i] = a.s;
    n[i] = a.n;
    const ctr = cellToLonLat(cells[i]!);
    if (ctr[0] < w || ctr[0] >= east || ctr[1] <= south || ctr[1] > north) continue;
    const ring = cellToBoundary(cells[i]!, { closedRing: true, segments: 1 });
    if ((p + ring.length) * 2 > coords.length || q + 3 * ring.length > tris.length) {
      const c2 = new Float64Array(coords.length * 2), t2 = new Uint32Array(tris.length * 2);
      c2.set(coords);
      t2.set(tris);
      coords = c2;
      tris = t2;
    }
    q = fan(ring, coords, p, tris, q);
    p += ring.length;
    rings.push(p);
    hrrrXY(ctr[0], ctr[1], wx, 2 * own.length);
    own.push(i);
  }
  const out = { cells, sum, n, own: Int32Array.from(own), wx: wx.slice(0, 2 * own.length), coords: coords.slice(0, p * 2), rings: Int32Array.from(rings), tris: tris.slice(0, q) };
  post({ type: "chunk", id, ...out }, [out.cells.buffer, out.sum.buffer, out.n.buffer, out.own.buffer, out.wx.buffer, out.coords.buffer, out.rings.buffer, out.tris.buffer]);
}

self.onmessage = (ev: MessageEvent<FineAsk>) => {
  const d = ev.data;
  if (d.type === "init") {
    H = d.head;
    root = d.root;
    t = d.t;
    lut = new Float32Array(BANDS * 256);
    for (let b = 0; b < BANDS; b++) {
      for (let v = -128; v < 128; v++) {
        const f = v / 127.5;
        lut[b * 256 + 128 + v] = (Math.sign(f) * f * f - H.mean[b]!) / H.scale[b]!;
      }
    }
    W1 = Float32Array.from(H.w1);
    B1 = Float32Array.from(H.b1);
    W2vol = Float32Array.from(H.w2.slice(0, H.hidden));
    b2vol = H.b2[0]!;
    post({ type: "ready", probeGap: probeGap() });
  } else {
    read(d.id, d.cy, d.cx, d.res).catch((e) => post({ type: "error", id: d.id, message: String(e?.message ?? e) }));
  }
};
