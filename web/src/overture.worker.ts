import earcut from "earcut";

/**
 * Overture geometries (WKB, as DuckDB hands them back) turned into flat arrays for the GeoArrow layers, off the main
 * thread: lines as coordinates and the first point of each line; polygons as their outer rings (holes are left
 * filled: this is context, not measurement) with the triangles already cut.
 */

export type GeomAsk = { id: number; kind: "lines" | "polygons"; values: Uint8Array; offsets: Int32Array };
export type GeomAnswer = { id: number; coords: Float64Array; starts: Int32Array; tris: Uint32Array };

class Grow {
  a = new Float64Array(1 << 16);
  n = 0;
  push2(x: number, y: number) {
    if (this.n + 2 > this.a.length) {
      const b = new Float64Array(this.a.length * 2);
      b.set(this.a);
      this.a = b;
    }
    this.a[this.n++] = x;
    this.a[this.n++] = y;
  }
}

function parse(d: GeomAsk): GeomAnswer {
  const view = new DataView(d.values.buffer, d.values.byteOffset, d.values.byteLength);
  const coords = new Grow();
  const starts: number[] = [0];
  const tris: number[] = [];
  let p = 0;
  /** One ring or line: its points appended to coords; returns how many. */
  const points = (le: boolean, dims: number, keep: boolean) => {
    const n = view.getUint32(p, le);
    p += 4;
    if (keep) for (let i = 0; i < n; i++) coords.push2(view.getFloat64(p + i * 8 * dims, le), view.getFloat64(p + i * 8 * dims + 8, le));
    p += n * 8 * dims;
    return n;
  };
  const geometry = () => {
    const le = view.getUint8(p) === 1;
    let type = view.getUint32(p + 1, le);
    p += 5;
    if (type & 0x20000000) p += 4; // an SRID, in the extended form
    const dims = 2 + (type & 0x80000000 || (type % 10000) >= 1000 ? 1 : 0) + (type & 0x40000000 || (type % 10000) >= 2000 ? 1 : 0);
    type = (type & 0xffff) % 1000;
    if (type === 1) p += 8 * dims;
    else if (type === 2) {
      const keep = d.kind === "lines";
      const n = points(le, dims, keep);
      if (keep && n > 1) starts.push(coords.n / 2);
      else if (keep) coords.n -= 2 * n;
    } else if (type === 3) {
      const rings = view.getUint32(p, le);
      p += 4;
      for (let r = 0; r < rings; r++) {
        const keep = d.kind === "polygons" && r === 0;
        const first = coords.n / 2;
        const n = points(le, dims, keep);
        if (!keep) continue;
        if (n < 4) {
          coords.n -= 2 * n;
          continue;
        }
        for (const t of earcut(coords.a.subarray(2 * first, coords.n))) tris.push(first + t);
        starts.push(coords.n / 2);
      }
    } else {
      // multi-geometries and collections: each part is a whole geometry
      const parts = view.getUint32(p, le);
      p += 4;
      for (let i = 0; i < parts; i++) geometry();
    }
  };
  for (let i = 0; i + 1 < d.offsets.length; i++) {
    p = d.offsets[i]!;
    if (d.offsets[i + 1]! > p) geometry();
  }
  return { id: d.id, coords: coords.a.slice(0, coords.n), starts: Int32Array.from(starts), tris: Uint32Array.from(tris) };
}

self.onmessage = (ev: MessageEvent<GeomAsk>) => {
  const r = parse(ev.data);
  (self as unknown as Worker).postMessage(r, [r.coords.buffer, r.starts.buffer, r.tris.buffer]);
};
