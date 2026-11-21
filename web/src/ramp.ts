/**
 * Viridis as 50 stops (aef-biomass ramp.ts, fire-knn aef/ramp.ts): matplotlib's path (dark purple, blue, teal, green,
 * yellow) with five darker stops below its first color toward near-black and three paler stops above its last toward
 * a pale lemon. Dark is the least fuel, bright the most.
 */
export const VIRIDIS_50 = [
  "#07020f", "#13021d", "#1f022b", "#2c0138", "#380146", "#440154", "#460a5d", "#471365", "#481b6d", "#482374",
  "#472c7a", "#46337f", "#443a83", "#424186", "#3e4989", "#3c508b", "#39568c", "#365d8d", "#32648e", "#306a8e",
  "#2d708e", "#2b758e", "#287c8e", "#26828e", "#24878e", "#228d8d", "#20938c", "#1f998a", "#1f9f88", "#20a486",
  "#25ab82", "#2ab07f", "#32b67a", "#3bbb75", "#48c16e", "#54c568", "#60ca60", "#6ece58", "#7fd34e", "#8ed645",
  "#9dd93b", "#addc30", "#c0df25", "#d0e11c", "#dfe318", "#efe51c", "#fde725", "#feef54", "#fef784", "#ffffb3",
];

/**
 * Inferno (matplotlib) from 14 percent up, with three paler stops above its pale yellow (aef-landcover-fosberg-cog
 * aef/ramp.ts): dark violet, crimson, orange, yellow; lightness rises the whole way. The risk is drawn on this one.
 */
export const INFERNO_50 = [
  "#260c51", "#2f0a5b", "#380962", "#400a67", "#470b6a", "#4f0d6c", "#57106e", "#5f136e", "#67166e", "#6d186e",
  "#751b6e", "#7d1e6d", "#85216b", "#8d2369", "#932667", "#9b2964", "#a32c61", "#ab2f5e", "#b1325a", "#b93556",
  "#c03a51", "#c73e4c", "#ce4347", "#d34743", "#d94d3d", "#df5337", "#e45a31", "#e9612b", "#ec6726", "#f06f20",
  "#f37819", "#f68013", "#f8870e", "#fa9008", "#fb9906", "#fca309", "#fcac11", "#fcb418", "#fbbe23", "#f9c72f",
  "#f7d13d", "#f5db4c", "#f3e35a", "#f1ec6d", "#f2f482", "#f6fa96", "#fcffa4", "#fdffb0", "#fdffbc", "#feffc8",
];

export const cssGradient = (stops: string[]) => `linear-gradient(90deg, ${stops.join(", ")})`;

/**
 * Where a volume (cover x height, m) sits on the ramp, 0..1: the volume head's own scale, log(0.02 + volume)
 * between 0 m and 50 m (models/fuel_volume.json vol_lo, vol_hi).
 */
const VOL_LO = Math.log(0.02);
const VOL_HI = Math.log(50);
export const volT = (v: number) => Math.min(1, Math.max(0, (Math.log(0.02 + v) - VOL_LO) / (VOL_HI - VOL_LO)));

const RGB = VIRIDIS_50.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)] as const);
/** The ramp's color at t (0..1) written into out at o as RGBA bytes. */
export function rampInto(out: Uint8Array, o: number, t: number) {
  const x = t * (RGB.length - 1), i = Math.min(RGB.length - 2, Math.floor(x)), u = x - i;
  const p = RGB[i]!, q = RGB[i + 1]!;
  out[o] = p[0] + (q[0] - p[0]) * u;
  out[o + 1] = p[1] + (q[1] - p[1]) * u;
  out[o + 2] = p[2] + (q[2] - p[2]) * u;
  out[o + 3] = 255;
}
