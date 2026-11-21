import { createReadStream, statSync } from "node:fs";
import { join, normalize } from "node:path";
import type { Connect, Plugin } from "vite";
import { defineConfig } from "vite";

/**
 * The Parquet files under public/data, served with byte ranges done properly. The dev server's own static handler
 * answers a suffix range ("bytes=-8", the last 8 bytes, which is how a Parquet reader finds the footer) with the
 * first bytes instead. Object stores and static hosts get this right; this is only for the local servers.
 */
function parquetRanges(): Plugin {
  const serve = (root: string): Connect.NextHandleFunction => (req, res, next) => {
    const path = decodeURIComponent((req.url ?? "").split("?")[0]!);
    if (!path.startsWith("/data/") || !path.endsWith(".parquet")) return next();
    const file = join(root, normalize(path));
    let size: number;
    try {
      size = statSync(file).size;
    } catch {
      return next();
    }
    res.setHeader("Accept-Ranges", "bytes");
    res.setHeader("Content-Type", "application/vnd.apache.parquet");
    res.setHeader("Cache-Control", "no-cache");
    const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
    let start = 0, end = size - 1;
    if (m && (m[1] || m[2])) {
      if (m[1] === "") start = Math.max(0, size - Number(m[2]));
      else {
        start = Number(m[1]);
        if (m[2] !== "") end = Math.min(end, Number(m[2]));
      }
      if (start > end) {
        res.statusCode = 416;
        res.setHeader("Content-Range", `bytes */${size}`);
        return res.end();
      }
      res.statusCode = 206;
      res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
    }
    res.setHeader("Content-Length", end - start + 1);
    if (req.method === "HEAD") return res.end();
    createReadStream(file, { start, end }).pipe(res);
  };
  return {
    name: "parquet-ranges",
    configureServer: (s) => void s.middlewares.use(serve(join(s.config.root, "public"))),
    configurePreviewServer: (s) => void s.middlewares.use(serve(join(s.config.root, s.config.build.outDir))),
  };
}

export default defineConfig({
  base: "./",
  plugins: [parquetRanges()],
  worker: { format: "es" },
  build: { target: "es2022" },
  server: { port: 5376, strictPort: true },
});
