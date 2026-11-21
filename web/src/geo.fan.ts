/**
 * A ring of points written as x, y into coords at point p and as a fan of triangles from its first corner into tris
 * at q (the cells are convex). The ring is closed: its last point is its first again. Returns the triangles' new end.
 */
export function fan(ring: ArrayLike<ArrayLike<number>>, coords: Float64Array, p: number, tris: Uint32Array, q: number) {
  const corners = ring.length - 1;
  for (let k = 1; k < corners - 1; k++) {
    tris[q++] = p;
    tris[q++] = p + k;
    tris[q++] = p + k + 1;
  }
  for (let k = 0; k < ring.length; k++) {
    coords[2 * (p + k)] = ring[k]![0]!;
    coords[2 * (p + k) + 1] = ring[k]![1]!;
  }
  return q;
}
