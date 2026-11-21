/**
 * Timings in the browser, not part of the map: the fuel Parquet read with parquet-wasm and with Haybarn-Wasm
 * (Query Farm's DuckDB-Wasm), and Overture's buildings, roads and infrastructure for one zoomed-in view with
 * Haybarn. Open /bench.html; the results are also left on window.__bench.
 */
import * as duckdb from "@haybarn/haybarn-wasm";
import ehWorker from "@haybarn/haybarn-wasm/dist/duckdb-browser-eh.worker.js?url";
import ehWasm from "@haybarn/haybarn-wasm/dist/duckdb-eh.wasm?url";
import { tableFromIPC } from "apache-arrow";
import initWasm, { ParquetFile } from "parquet-wasm/esm";
import pqWasm from "parquet-wasm/esm/parquet_wasm_bg.wasm?url";

const out: Record<string, unknown> = {};
const el = document.getElementById("out")!;
const note = (k: string, v: unknown) => {
  out[k] = v;
  el.textContent = JSON.stringify(out, null, 1);
};
const ms = (t: number) => Math.round(performance.now() - t);
const base = new URL("./data/", location.href).href;
const RES = 14;
/** Two runs of 29 row groups, what a zoom 8 view reads. */
const RUNS: [number, number][] = [[30, 58], [60, 88]];

async function main() {
  // parquet-wasm, as the map reads now: four row groups at a time
  let t = performance.now();
  await initWasm({ module_or_path: pqWasm });
  note("parquet-wasm start ms", ms(t));
  t = performance.now();
  const file = await ParquetFile.fromUrl(`${base}fuel_r${RES}.parquet`);
  note("parquet-wasm open file ms", ms(t));
  for (const [a, b] of RUNS) {
    t = performance.now();
    const rgs = Array.from({ length: b - a + 1 }, (_, i) => a + i);
    let rows = 0, next = 0;
    await Promise.all(Array.from({ length: 4 }, async () => {
      while (next < rgs.length) {
        const got = tableFromIPC((await file.read({ rowGroups: [rgs[next++]!] })).intoIPCStream()).numRows;
        rows += got;
      }
    }));
    note(`parquet-wasm row groups ${a}-${b}`, { ms: ms(t), rows });
  }

  // Haybarn
  t = performance.now();
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), new Worker(ehWorker));
  await db.instantiate(ehWasm);
  // byte ranges only: as it comes, this build downloads every remote file whole
  await db.open({ path: ":memory:", filesystem: { forceFullHTTPReads: false, reliableHeadRequests: true, allowFullHTTPReads: false } });
  const con = await db.connect();
  note("haybarn start ms", ms(t));
  t = performance.now();
  const ix = await con.query(`SELECT rg, lo, hi FROM '${base}fuel_index.parquet' WHERE res = ${RES} ORDER BY rg`);
  note("haybarn first query (index, loads the Parquet reader) ms", ms(t));
  const lo = ix.getChild("lo")!.toArray() as BigUint64Array, hi = ix.getChild("hi")!.toArray() as BigUint64Array;
  for (const [a, b] of RUNS) {
    t = performance.now();
    const r = await con.query(`SELECT cell, vol, n FROM '${base}fuel_r${RES}.parquet' WHERE cell BETWEEN ${lo[a]} AND ${hi[b]}`);
    note(`haybarn row groups ${a}-${b}`, { ms: ms(t), rows: r.numRows });
  }

  // Overture for one zoomed-in view (Georgetown, California), then the view next to it
  const rel = (await (await fetch("https://stac.overturemaps.org/catalog.json")).json()).latest as string;
  note("overture release", rel);
  // Overture names each file's bounding box in its catalog (one fetch per theme) and the bucket lists its files,
  // so only the file or two under the view is opened; DuckDB's own row group statistics do the rest
  const themes = [["buildings", "building"], ["transportation", "segment"], ["base", "infrastructure"]] as const;
  const files = new Map<string, { urls: string[]; boxes: number[][] }>();
  for (const [theme, type] of themes) {
    t = performance.now();
    const col = await (await fetch(`https://stac.overturemaps.org/${rel}/${theme}/${type}/collection.json`)).json();
    const list = await (await fetch(`https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/?list-type=2&prefix=release/${rel}/theme=${theme}/type=${type}/`)).text();
    const urls = [...list.matchAll(/<Key>([^<]+\.parquet)<\/Key>/g)].map((m) => `https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/${m[1]}`).sort();
    files.set(type, { urls, boxes: (col.extent.spatial.bbox as number[][]).slice(1) });
    note(`overture ${type} catalog`, { ms: ms(t), files: urls.length, boxes: col.extent.spatial.bbox.length - 1 });
  }
  for (const [name, w, s, e, n] of [["Berkeley", -122.31, 37.84, -122.23, 37.9], ["Georgetown", -120.77, 38.86, -120.63, 38.94], ["next to Georgetown", -120.63, 38.86, -120.49, 38.94]] as const) {
    for (const [, type] of themes) {
      const f = files.get(type)!;
      const hit = f.urls.filter((_, i) => f.boxes[i] && f.boxes[i]![0]! < e && f.boxes[i]![2]! > w && f.boxes[i]![1]! < n && f.boxes[i]![3]! > s);
      t = performance.now();
      try {
        const r = await con.query(`SELECT id, geometry FROM read_parquet([${hit.map((u) => `'${u}'`).join(", ")}])
          WHERE bbox.xmin < ${e} AND bbox.xmax > ${w} AND bbox.ymin < ${n} AND bbox.ymax > ${s}`);
        note(`overture ${name} ${type}`, { ms: ms(t), rows: r.numRows, files: hit.length });
      } catch (err) {
        note(`overture ${name} ${type}`, { ms: ms(t), files: hit.length, error: String((err as Error)?.message ?? err).slice(0, 300) });
      }
    }
  }
  note("done", true);
}
main().catch((e) => note("failed", String(e?.message ?? e)));
(globalThis as unknown as { __bench: unknown }).__bench = out;
