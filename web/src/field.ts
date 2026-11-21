import { BitmapLayer } from "@deck.gl/layers";
import type { Device, Texture } from "@luma.gl/core";

import type { FieldAnswer, FieldAsk } from "./field.worker";
import { SIDE, TILE } from "./field.worker";
import FieldWorker from "./field.worker.ts?worker";
import { riskModule, riskProps, TILE_INJECT } from "./risk";

/**
 * The fuel below the zoom where the embeddings are read: map tiles made from the A5 cells (field.worker.ts), drawn
 * as textures. The cells are chosen about one to a texel, so what shows is a smooth picture and no cell outlines,
 * and a full window is a few dozen textured squares instead of a million polygons.
 */

const EARTH_M2 = 5.1007e14;
/** A cell's width in m: the side of a square of its area. */
export const cellM = (res: number) => Math.sqrt(EARTH_M2 / (60 * 4 ** (res - 1)));
/** Tiles kept (270 KB each, and as much again on the GPU) before the ones seen longest ago are let go. */
const MAX_TILES = 420;
const WORKERS = 3;
const LINEAR = { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" } as const;

const texelM = (z: number, lat: number) => (40_075_017 * Math.cos((lat * Math.PI) / 180)) / (TILE * 2 ** z);
const latOf = (fy: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * fy))) * 180) / Math.PI;
const lonOf = (fx: number) => fx * 360 - 180;
const fxOf = (lon: number) => (lon + 180) / 360;
const fyOf = (lat: number) => (1 - Math.asinh(Math.tan((lat * Math.PI) / 180)) / Math.PI) / 2;

type Tile = { z: number; x: number; y: number; res: number; data: Uint8Array | null; tex: Texture | null; seen: number };
/** What to draw in one tile's square: the tile itself, or part of a coarser one (u, v, size as fractions of it) until it comes. */
type Draw = { id: string; z: number; x: number; y: number; src: Tile; sub: [number, number, number] };

class FuelTileLayer extends BitmapLayer<{ tile: Tile; sub: [number, number, number]; merc: [number, number, number, number] }> {
  static layerName = "FuelTileLayer";
  getShaders() {
    const s = super.getShaders();
    return { ...s, modules: [...s.modules, riskModule], inject: TILE_INJECT };
  }
  draw(opts: never) {
    const { tile, sub, merc } = this.props;
    const device = this.context.device;
    if (!tile.tex) return;
    const model = this.state.model as { shaderInputs: { setProps(p: unknown): void } } | undefined;
    // the tile's inner square inside its one texel border, then the part of it this square shows
    const pad = 1 / SIDE, inner = TILE / SIDE;
    model?.shaderInputs.setProps(riskProps(device, { fuelTex: tile.tex, merc, uv: [pad + sub[0] * inner, pad + sub[1] * inner, sub[2] * inner, sub[2] * inner] }));
    super.draw(opts);
  }
}

export function createField(onChange: () => void, onError: (m: string) => void) {
  const workers = Array.from({ length: WORKERS }, () => new FieldWorker());
  const tiles = new Map<string, Tile>();
  const failed = new Set<string>();
  let resList: number[] = [];
  let want: { z: number; x: number; y: number; res: number; key: string }[] = [];
  let pendingView: { zoom: number; bbox: [number, number, number, number] } | null = null;
  let tick = 0;
  let wantRes = 0;
  /** One texture for every tile with no ground. */
  let empty: Texture | null = null;
  const key = (z: number, x: number, y: number) => `${z}/${x}/${y}`;

  for (const w of workers) {
    w.onmessage = (ev: MessageEvent<FieldAnswer>) => {
      const d = ev.data;
      if (d.type === "index") {
        if (!resList.length) {
          resList = d.res;
          if (pendingView) view(pendingView.zoom, pendingView.bbox);
        }
      } else if (d.type === "tile") {
        tiles.set(key(d.z, d.x, d.y), { z: d.z, x: d.x, y: d.y, res: d.res, data: d.data, tex: null, seen: tick });
        trim();
        onChange();
      } else {
        if (d.z != null) failed.add(key(d.z, d.x!, d.y!));
        onError(d.message);
        onChange();
      }
    };
    w.postMessage({ type: "init", base: new URL("./", location.href).href } satisfies FieldAsk);
  }

  function trim() {
    if (tiles.size <= MAX_TILES) return;
    const keep = new Set(want.map((t) => t.key));
    const old = [...tiles].filter(([k]) => !keep.has(k)).sort((a, b) => a[1].seen - b[1].seen);
    for (const [k, t] of old) {
      if (tiles.size <= MAX_TILES) break;
      if (t.data) t.tex?.destroy();
      tiles.delete(k);
    }
  }

  /** The resolution whose cells are about a texel wide at a tile zoom and latitude, among the ones baked. */
  function resFor(z: number, lat: number) {
    const m = texelM(z, lat);
    let best = resList[0]!;
    for (const r of resList) if (cellM(r) >= 0.71 * m) best = r;
    return best;
  }
  /** The finest tile zoom worth making: past it the finest cells would be wider than a texel and show their shape. */
  function zMax(lat: number) {
    const finest = cellM(resList[resList.length - 1]!);
    let z = 3;
    while (z < 14 && texelM(z + 1, lat) >= 0.71 * finest) z++;
    return z;
  }

  function view(zoom: number, bbox: [number, number, number, number]) {
    pendingView = { zoom, bbox };
    if (!resList.length) return;
    tick++;
    const mid = (bbox[1] + bbox[3]) / 2;
    // a texel is about one device pixel on a 2x display (half a screen point), so the picture is sharp, not stretched
    const z = Math.max(3, Math.min(zMax(mid), Math.floor(zoom + 0.5) + 1));
    const n = 2 ** z;
    const x0 = Math.floor(fxOf(bbox[0]) * n), x1 = Math.floor(fxOf(bbox[2]) * n);
    const y0 = Math.max(0, Math.floor(fyOf(Math.min(85, bbox[3])) * n)), y1 = Math.min(n - 1, Math.floor(fyOf(Math.max(-85, bbox[1])) * n));
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    const list: (typeof want)[number][] = [];
    for (let y = y0; y <= y1; y++) {
      const res = resFor(z, latOf((y + 0.5) / n));
      for (let x = x0; x <= x1; x++) list.push({ z, x: ((x % n) + n) % n, y, res, key: key(z, ((x % n) + n) % n, y) });
    }
    // from the middle of the view outward
    want = list.sort((a, b) => (a.x - mx) ** 2 + (a.y - my) ** 2 - ((b.x - mx) ** 2 + (b.y - my) ** 2));
    wantRes = resFor(z, mid);
    for (const t of want) {
      const have = tiles.get(t.key);
      if (have) have.seen = tick;
    }
    // neighbors share row groups, so a block of 4 x 4 tiles goes to one worker and is read once
    const jobs: { z: number; x: number; y: number; res: number }[][] = workers.map(() => []);
    for (const t of want) if (!tiles.has(t.key) && !failed.has(t.key)) jobs[((t.x >> 2) * 7 + (t.y >> 2) * 13) % WORKERS]!.push({ z: t.z, x: t.x, y: t.y, res: t.res });
    workers.forEach((w, i) => w.postMessage({ type: "want", jobs: jobs[i]! } satisfies FieldAsk));
    onChange();
  }

  /** What stands in a wanted tile's square: itself, or the part of the nearest coarser tile in hand that covers it. */
  function standIn(t: (typeof want)[number]): Draw[] {
    const own = tiles.get(t.key);
    // a tile with no ground is still drawn: the weather shows there
    if (own) return [{ id: t.key, z: t.z, x: t.x, y: t.y, src: own, sub: [0, 0, 1] }];
    for (let up = 1; up <= 6 && t.z - up >= 0; up++) {
      const p = tiles.get(key(t.z - up, t.x >> up, t.y >> up));
      if (!p) continue;
      const k = 2 ** up;
      return [{ id: `${t.key}<${up}`, z: t.z, x: t.x, y: t.y, src: p, sub: [(t.x % k) / k, (t.y % k) / k, 1 / k] }];
    }
    // zooming out: the four finer tiles inside it that are in hand
    const out: Draw[] = [];
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const c = tiles.get(key(t.z + 1, 2 * t.x + dx, 2 * t.y + dy));
      if (c) out.push({ id: key(c.z, c.x, c.y), z: c.z, x: c.x, y: c.y, src: c, sub: [0, 0, 1] });
    }
    return out;
  }

  return {
    /** Ask for the tiles of a view (west, south, east, north). */
    view,
    /** The A5 resolution the tiles on screen are made from. */
    get res() {
      return wantRes;
    },
    progress() {
      const got = want.filter((t) => tiles.has(t.key) || failed.has(t.key)).length;
      return { got, total: want.length, ready: resList.length > 0, done: resList.length > 0 && got === want.length };
    },
    /** The fuel under a point, 0..1 on the ramp's scale, from the tile on screen; null where there is no ground or no tile yet. */
    at(lon: number, lat: number) {
      const z = want[0]?.z;
      if (z == null) return null;
      const n = 2 ** z, fx = fxOf(lon) * n, fy = fyOf(lat) * n;
      const t = tiles.get(key(z, Math.floor(fx), Math.floor(fy)));
      if (!t?.data) return null;
      const o = 4 * ((1 + Math.floor((fy - t.y) * TILE)) * SIDE + 1 + Math.floor((fx - t.x) * TILE));
      return t.data[o + 3] ? { t: t.data[o]! / 255, res: t.res } : null;
    },
    layers(beforeId: string | undefined, device: Device | undefined) {
      const out: FuelTileLayer[] = [];
      if (!device) return out;
      for (const w of want) {
        for (const d of standIn(w)) {
          const n = 2 ** d.z;
          d.src.tex ??= d.src.data
            ? device.createTexture({ width: SIDE, height: SIDE, format: "rgba8unorm", data: d.src.data, mipLevels: 1, sampler: LINEAR })
            : (empty ??= device.createTexture({ width: 1, height: 1, format: "rgba8unorm", data: new Uint8Array(4), mipLevels: 1, sampler: LINEAR }));
          const merc: [number, number, number, number] = [d.x / n, d.y / n, (d.x + 1) / n, (d.y + 1) / n];
          out.push(
            new FuelTileLayer({
              id: `fuel-${d.id}`,
              image: d.src.tex as never,
              bounds: [lonOf(merc[0]), latOf(merc[3]), lonOf(merc[2]), latOf(merc[1])],
              tile: d.src,
              sub: d.sub,
              merc,
              pickable: false,
              // read by the MapLibre overlay: the layer draws under the labels
              beforeId,
            } as never),
          );
        }
      }
      return out;
    },
  };
}
