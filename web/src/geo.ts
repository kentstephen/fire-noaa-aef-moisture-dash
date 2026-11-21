import { GeoArrowSolidPolygonLayer } from "@geoarrow/deck.gl-layers";
import { Field, FixedSizeList, Float64, List, makeData, RecordBatch, Schema, Struct, Uint8 } from "apache-arrow";

/** Pentagons as GeoArrow, for both kinds of cells on the map: the baked ones (cells.ts) and the ones made here (fine.ts). */

const XY = new FixedSizeList(2, new Field("xy", new Float64(), false));
const RING = new List(new Field("vertices", XY, false));
const POLY = new List(new Field("rings", RING, false));
const RGBA = new FixedSizeList(4, new Field("c", new Uint8(), false));
const F_POLY = new Field("geometry", POLY, false, new Map([["ARROW:extension:name", "geoarrow.polygon"]]));
const SCHEMA = new Schema([F_POLY]);

/**
 * Polygons as one GeoArrow record batch, no copies (the arrays are the buffers): coords is x, y of every ring point,
 * rings the first point of each polygon's one ring and the total at the end.
 */
export function polygonBatch(coords: Float64Array, rings: Int32Array) {
  const count = rings.length - 1;
  const points = coords.length / 2;
  const xy = makeData({ type: XY, length: points, nullCount: 0, child: makeData({ type: new Float64(), length: points * 2, nullCount: 0, data: coords }) });
  const ring = makeData({ type: RING, length: count, nullCount: 0, valueOffsets: rings, child: xy });
  const one = new Int32Array(count + 1);
  for (let i = 0; i <= count; i++) one[i] = i;
  const poly = makeData({ type: POLY, length: count, nullCount: 0, valueOffsets: one, child: ring });
  const batch = new RecordBatch(SCHEMA, makeData({ type: new Struct([F_POLY]), length: count, nullCount: 0, children: [poly] }));
  return { batch, poly: poly as unknown };
}

/** One RGBA per polygon as an Arrow column the layer takes for its fill. */
export const colorData = (rgba: Uint8Array): unknown =>
  makeData({ type: RGBA, length: rgba.length / 4, nullCount: 0, child: makeData({ type: new Uint8(), length: rgba.length, nullCount: 0, data: rgba }) });

/** Kyle Barron's polygon layer, given the triangles (made in a worker) instead of cutting them on this thread. */
export class CellLayer extends GeoArrowSolidPolygonLayer<{ tris: Uint32Array; beforeId?: string }> {
  static layerName = "CellLayer";
  async _updateEarcut() {
    return this.props.tris;
  }
}

export function cellLayer(id: string, g: { batch: RecordBatch; poly: unknown; color: unknown; tris: Uint32Array }, beforeId: string | undefined) {
  return new CellLayer({
    id,
    data: g.batch,
    getPolygon: g.poly as never,
    getFillColor: g.color as never,
    tris: g.tris,
    // no triangulation worker fetched from a CDN: the triangles are already made
    earcutWorkerUrl: null,
    filled: true,
    extruded: false,
    pickable: false,
    // read by the MapLibre overlay: the layer draws under the labels
    beforeId,
  });
}
