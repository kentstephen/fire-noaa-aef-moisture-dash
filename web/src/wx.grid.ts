/**
 * The HRRR grid (3 km, Lambert conformal on a sphere) and its blocks: the analysis store keeps 45 x 45 pixel chunks,
 * about 135 km square, and the weather is read one such block at a time.
 */
export const HRRR = { x0: -2699020.142521929, y0: 1588193.847443335, d: 3000, w: 1799, h: 1059 };
export const BLOCK = 45;
const BLOCK_COLS = Math.ceil(HRRR.w / BLOCK);

const RAD = Math.PI / 180, R = 6371229, LAT0 = 38.5 * RAD, LON0 = -97.5 * RAD;
const N = Math.sin(LAT0);
const F = (Math.cos(LAT0) * Math.tan(Math.PI / 4 + LAT0 / 2) ** N) / N;
const RHO0 = (R * F) / Math.tan(Math.PI / 4 + LAT0 / 2) ** N;

/**
 * A point's place on the HRRR grid as column and row, in pixels and fractions of one, counted from pixel centers
 * (a whole number is the middle of a pixel), written into out at o. Off the grid: NaN.
 */
export function hrrrXY(lon: number, lat: number, out: Float32Array, o: number) {
  const rho = (R * F) / Math.tan(Math.PI / 4 + (lat * RAD) / 2) ** N;
  const th = N * (lon * RAD - LON0);
  const col = (rho * Math.sin(th) - HRRR.x0) / HRRR.d - 0.5;
  const row = (HRRR.y0 - (RHO0 - rho * Math.cos(th))) / HRRR.d - 0.5;
  const ok = col >= 0 && col <= HRRR.w - 1 && row >= 0 && row <= HRRR.h - 1;
  out[o] = ok ? col : NaN;
  out[o + 1] = ok ? row : NaN;
}

/** The block a pixel (column, row) is in, and a block's row and column among the blocks. */
export const blockAt = (col: number, row: number) => Math.floor(row / BLOCK) * BLOCK_COLS + Math.floor(col / BLOCK);
export const blockRowCol = (b: number): [number, number] => [Math.floor(b / BLOCK_COLS), b % BLOCK_COLS];
