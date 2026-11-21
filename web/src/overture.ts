import type { RecordBatch } from "apache-arrow";

import { lineBatch, lineLayer } from "./geo";
import type { GeomAnswer, GeomAsk } from "./overture.worker";
import GeomWorker from "./overture.worker.ts?worker";

/**
 * What is on the ground, as context when zoomed in to where the embeddings are read: Overture Maps' roads, power
 * lines and building outlines, in one neutral color each, not colored by the fuel. Read in the browser with DuckDB
 * (Haybarn-Wasm) straight from Overture's GeoParquet on S3: Overture's catalog names each file's bounding box, so
 * only the file or two under the view is opened, and DuckDB's row group statistics find the rows.
 *
 * DuckDB is a large download, so it is fetched only on the way in (WARM_ZOOM), never for the statewide map.
 */

/** Roads and power lines from here in (where the embeddings start), buildings from BUILDING_ZOOM. */
export const CONTEXT_ZOOM = 12.5;
const BUILDING_ZOOM = 14;
/** DuckDB and Overture's catalog are fetched from this zoom, to be in hand by CONTEXT_ZOOM. */
const WARM_ZOOM = 11.5;
const BUCKET = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com";
const STAC = "https://stac.overturemaps.org";
/** More than this in one view is not read (a whole city of buildings). */
const MAX_ROWS = 250_000;

type Box = [number, number, number, number];
type Kind = "roads" | "power" | "buildings";
const SOURCES: Record<Kind, { theme: string; type: string; where: string; geom: "lines" | "outlines" }> = {
  roads: { theme: "transportation", type: "segment", where: "subtype = 'road'", geom: "lines" },
  power: { theme: "base", type: "infrastructure", where: "class IN ('power_line', 'minor_line')", geom: "lines" },
  buildings: { theme: "buildings", type: "building", where: "true", geom: "outlines" },
};
type Drawn = { box: Box; rows: number; batch: RecordBatch; line: unknown } | null;
type Query = (sql: string) => Promise<{ numRows: number; getChild(name: string): { data: { values: Uint8Array; valueOffsets: Int32Array; length: number; offset: number }[] } | null }>;

const inside = (a: Box, b: Box) => a[0] >= b[0] && a[1] >= b[1] && a[2] <= b[2] && a[3] <= b[3];

export function createOverture(onChange: () => void, onError: (m: string) => void) {
  let starting: Promise<{ query: Query; files: Record<Kind, { urls: string[]; boxes: number[][] }>; stop: () => void }> | null = null;
  /** One query at a time: DuckDB in WebAssembly runs out of memory when several large reads overlap. */
  let line: Promise<unknown> = Promise.resolve();
  const inTurn = <T>(fn: () => Promise<T>): Promise<T> => {
    const p = line.then(fn, fn);
    line = p.catch(() => {});
    return p;
  };
  const worker = new GeomWorker();
  const waiting = new Map<number, (a: GeomAnswer) => void>();
  worker.onmessage = (ev: MessageEvent<GeomAnswer>) => {
    waiting.get(ev.data.id)?.(ev.data);
    waiting.delete(ev.data.id);
  };
  let nextId = 1;
  const shapes = (a: Omit<GeomAsk, "id">) =>
    new Promise<GeomAnswer>((ok) => {
      const id = nextId++;
      waiting.set(id, ok);
      worker.postMessage({ id, ...a } satisfies GeomAsk, [a.values.buffer, a.offsets.buffer]);
    });

  const drawn: Record<Kind, Drawn> = { roads: null, power: null, buildings: null };
  const busy: Record<Kind, boolean> = { roads: false, power: false, buildings: false };
  const note: Record<Kind, string> = { roads: "", power: "", buildings: "" };
  const gaveUp: Record<Kind, string> = { roads: "", power: "", buildings: "" };
  let view: { zoom: number; box: Box } | null = null;
  let active = false;

  function start() {
    starting ??= (async () => {
      const duckdb = await import("@haybarn/haybarn-wasm");
      const [{ default: workerUrl }, { default: wasmUrl }] = await Promise.all([
        import("@haybarn/haybarn-wasm/dist/duckdb-browser-eh.worker.js?url"),
        import("@haybarn/haybarn-wasm/dist/duckdb-eh.wasm?url"),
      ]);
      const engine = new Worker(workerUrl);
      const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), engine);
      await db.instantiate(wasmUrl);
      // byte ranges only: as it comes, this build downloads every remote file whole (Overture's are 700 to 800 MB each)
      await db.open({ path: ":memory:", filesystem: { forceFullHTTPReads: false, reliableHeadRequests: true, allowFullHTTPReads: false } });
      const con = await db.connect();
      const release = (await (await fetch(`${STAC}/catalog.json`)).json()).latest as string;
      const files = {} as Record<Kind, { urls: string[]; boxes: number[][] }>;
      await Promise.all(
        (Object.keys(SOURCES) as Kind[]).map(async (k) => {
          const { theme, type } = SOURCES[k];
          const [col, list] = await Promise.all([
            fetch(`${STAC}/${release}/${theme}/${type}/collection.json`).then((r) => r.json()),
            fetch(`${BUCKET}/?list-type=2&prefix=release/${release}/theme=${theme}/type=${type}/`).then((r) => r.text()),
          ]);
          const urls = [...list.matchAll(/<Key>([^<]+\.parquet)<\/Key>/g)].map((m) => `${BUCKET}/${m[1]}`).sort();
          // the catalog's boxes: the whole collection's first, then one per file in file order
          const boxes = (col.extent.spatial.bbox as number[][]).slice(1);
          if (urls.length !== boxes.length) throw new Error(`Overture ${type}: ${urls.length} files, ${boxes.length} boxes`);
          files[k] = { urls, boxes };
        }),
      );
      return { query: (sql: string) => inTurn(() => con.query(sql) as unknown as ReturnType<Query>), files, stop: () => engine.terminate() };
    })();
    starting.catch((e) => {
      starting = null;
      onError(`Overture: ${e?.message ?? e}`);
    });
    return starting;
  }

  async function read(kind: Kind) {
    if (busy[kind] || !view) return;
    const wanted = kind === "buildings" ? view.zoom >= BUILDING_ZOOM : true;
    if (!wanted || (drawn[kind] && inside(view.box, drawn[kind]!.box))) return;
    busy[kind] = true;
    onChange();
    try {
      const { query, files } = await start();
      // a third of the view more on every side, so a small pan needs no new read
      const [w0, s0, e0, n0] = view.box;
      const mx = (e0 - w0) / 3, my = (n0 - s0) / 3;
      const box: Box = [w0 - mx, s0 - my, e0 + mx, n0 + my];
      const [w, s, e, n] = box;
      const f = files[kind];
      const urls = f.urls.filter((_, i) => f.boxes[i]![0]! < e && f.boxes[i]![2]! > w && f.boxes[i]![1]! < n && f.boxes[i]![3]! > s);
      const src = SOURCES[kind];
      let coords: Float64Array = new Float64Array(0), starts: Int32Array = new Int32Array([0]), rows = 0;
      if (urls.length) {
        const t = await query(`SELECT geometry FROM read_parquet([${urls.map((u) => `'${u}'`).join(", ")}])
          WHERE bbox.xmin < ${e} AND bbox.xmax > ${w} AND bbox.ymin < ${n} AND bbox.ymax > ${s} AND ${src.where} LIMIT ${MAX_ROWS + 1}`);
        rows = t.numRows;
        // the WKB of every row as one run of bytes and where each row starts in it
        const parts = t.getChild("geometry")!.data;
        let bytes = 0;
        for (const d of parts) bytes += d.valueOffsets[d.offset + d.length]! - d.valueOffsets[d.offset]!;
        const values = new Uint8Array(bytes), offsets = new Int32Array(rows + 1);
        let at = 0, row = 0;
        for (const d of parts) {
          const a = d.valueOffsets[d.offset]!;
          values.set(d.values.subarray(a, d.valueOffsets[d.offset + d.length]!), at);
          for (let i = 0; i < d.length; i++) offsets[row++] = at + d.valueOffsets[d.offset + i]! - a;
          at += d.valueOffsets[d.offset + d.length]! - a;
        }
        offsets[rows] = at;
        ({ coords, starts } = await shapes({ kind: src.geom, values, offsets }));
      }
      note[kind] = rows > MAX_ROWS ? `more than ${MAX_ROWS.toLocaleString("en-US")} ${kind} here, not all drawn` : "";
      drawn[kind] = { box, rows, ...lineBatch(coords, starts) };
    } catch (e) {
      const m = String((e as Error)?.message ?? e);
      onError(`Overture ${kind}: ${m}`);
      // a WebAssembly fault leaves the engine unusable: drop it, the next read starts a new one
      if (/memory|unreachable|RuntimeError|out of bounds/i.test(m)) {
        void starting?.then((s) => s.stop()).catch(() => {});
        starting = null;
      }
      gaveUp[kind] = view ? view.box.join() : "";
    }
    busy[kind] = false;
    onChange();
    // the view may have moved on while this was read (a view that just failed is not tried again until it moves)
    if (view && active && gaveUp[kind] !== view.box.join()) void read(kind);
  }

  return {
    /** Tell it the view (west, south, east, north); below CONTEXT_ZOOM nothing is read or drawn. */
    view(zoom: number, box: Box) {
      active = zoom >= CONTEXT_ZOOM;
      view = { zoom, box };
      if (zoom >= WARM_ZOOM) void start().catch(() => {});
      if (!active) return;
      for (const k of Object.keys(SOURCES) as Kind[]) void read(k);
    },
    get active() {
      return active;
    },
    /** What is being read now, and anything left out. */
    status() {
      const reading = (Object.keys(busy) as Kind[]).filter((k) => busy[k]);
      return { reading: active ? reading : [], notes: active ? Object.values(note).filter(Boolean) : [], counts: { roads: drawn.roads?.rows ?? 0, power: drawn.power?.rows ?? 0, buildings: drawn.buildings?.rows ?? 0 } };
    },
    layers(beforeId: string | undefined) {
      if (!active || !view) return [];
      const out = [];
      const b = drawn.buildings, r = drawn.roads, p = drawn.power;
      if (b && b.rows && view.zoom >= BUILDING_ZOOM) out.push(lineLayer("overture-buildings", b, [240, 242, 244, 200], 1, beforeId));
      if (r && r.rows) out.push(lineLayer("overture-roads", r, [240, 242, 244, 130], 1, beforeId));
      if (p && p.rows) out.push(lineLayer("overture-power", p, [255, 255, 255, 235], 2.5, beforeId));
      return out;
    },
  };
}
