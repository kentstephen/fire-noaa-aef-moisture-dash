import type { Device, Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";

import { INFERNO_50, VIRIDIS_50 } from "./ramp";
import { HRRR } from "./wx.grid";

/**
 * The one picture every layer draws, per screen pixel, the way aef-landcover-fosberg-cog draws it (its aef/render.ts):
 * the ground's own score and the hour's weather score, both in logits, summed and put through a sigmoid.
 *
 * - Here the ground's score is its fuel: the volume head's 0..1 stretched over GROUND_LO..GROUND_HI, the span of that
 *   app's ground scores.
 * - The weather is drawn in gray under everything, ground or not: dark three logits under the gate, pale one over it.
 * - The ground is faint and gray (the ramp's own lightness, colored by the ground alone) where the weather is under
 *   the gate, and goes opaque and takes its color (ground plus weather) over half a logit as the weather crosses it.
 * - The ramp runs over the probability from pHide to pMax.
 * The weather score read is the hour's own, or the built-up one (look.sustained). With the weather off the ramp shows
 * the fuel alone, on its own ramp.
 */

/** The weather score's range in the textures (a byte holds 1 to 255 over it). */
export const TERM_LO = -6, TERM_HI = 6;
const GROUND_LO = -2.43, GROUND_HI = 2.8;

export const look = {
  /** 1 = risk (fuel with the weather), 0 = the fuel alone. */
  weather: 1,
  /** 1 = the weather built up over days decides what is lit, 0 = the hour's own weather. */
  sustained: 0,
  /** The place on the weather's time axis, in hours and fractions of one. */
  frame: 0,
  /** The weather score at which the ground goes opaque and takes its color. */
  gate: 1.0,
  /** The ramp: hidden below this probability, its top at this one. */
  pHide: 0.5,
  pMax: 0.95,
  /** Opacity of the ground where the weather is under the gate, and of the weather field under it. */
  standing: 0.25,
  field: 0.35,
};

/**
 * What the weather hands the shader: the score of each hour (one layer per hour), the built-up score (a layer every
 * slowStep hours), the mask of pixels in hand, and where they sit on the HRRR grid.
 */
export type WxTex = { tex: Texture; slow: Texture; slowStep: number; mask: Texture; col0: number; row0: number; w: number; h: number; frames: number };
let wxTex: WxTex | null = null;
let rampTex: Texture | null = null;
let fuelRampTex: Texture | null = null;
let blank: Texture | null = null;
let blankArr: Texture | null = null;
export const setWxTex = (t: WxTex | null) => {
  wxTex?.tex.destroy();
  wxTex?.slow.destroy();
  wxTex?.mask.destroy();
  wxTex = t;
};

const LINEAR = { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" } as const;

function stripe(device: Device, stops: string[]) {
  const n = 256, data = new Uint8Array(n * 4);
  const rgb = stops.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(x)), u = x - k;
    for (let c = 0; c < 3; c++) data[4 * i + c] = rgb[k]![c]! + (rgb[k + 1]![c]! - rgb[k]![c]!) * u;
    data[4 * i + 3] = 255;
  }
  return device.createTexture({ width: n, height: 1, format: "rgba8unorm", data, mipLevels: 1, sampler: LINEAR });
}
function ramp(device: Device) {
  if (rampTex) return rampTex;
  rampTex = stripe(device, INFERNO_50);
  fuelRampTex = stripe(device, VIRIDIS_50);
  blank = device.createTexture({ width: 1, height: 1, format: "r8unorm", data: new Uint8Array(1), mipLevels: 1, sampler: LINEAR });
  blankArr = device.createTexture({ dimension: "2d-array", width: 1, height: 1, depth: 1, format: "r8unorm", data: new Uint8Array(1), mipLevels: 1, sampler: LINEAR });
  return rampTex;
}

export type RiskProps = { fuelTex?: Texture; merc?: [number, number, number, number]; uv?: [number, number, number, number] };

/** The module's inputs for one draw: the shared look and weather, plus a tile's own fuel texture and place. */
export function riskProps(device: Device, own: RiskProps = {}) {
  const r = ramp(device);
  const on = look.weather > 0.5 && wxTex ? 1 : 0;
  const slow = on && look.sustained > 0.5;
  const top = wxTex ? (slow ? wxTex.frames / wxTex.slowStep : wxTex.frames) - 1 : 0;
  return {
    risk: {
      useWeather: on,
      frame: wxTex ? Math.min(top, Math.max(0, slow ? look.frame / wxTex.slowStep : look.frame)) : 0,
      gate: look.gate,
      pHide: look.pHide,
      pMax: look.pMax,
      standing: look.standing,
      field: look.field,
      grid: wxTex ? [wxTex.col0, wxTex.row0, wxTex.w, wxTex.h] : [0, 0, 1, 1],
      merc: own.merc ?? [0, 0, 1, 1],
      uv: own.uv ?? [0, 0, 1, 1],
      rampTex: r,
      fuelRampTex: fuelRampTex!,
      fuelTex: own.fuelTex ?? r,
      wxTex: (slow ? wxTex?.slow : wxTex?.tex) ?? blankArr!,
      maskTex: wxTex?.mask ?? blank!,
    },
  };
}

const RAD = Math.PI / 180, R = 6371229, LAT0 = 38.5 * RAD;
const N = Math.sin(LAT0);
const F = (Math.cos(LAT0) * Math.tan(Math.PI / 4 + LAT0 / 2) ** N) / N;
const RHO0 = (R * F) / Math.tan(Math.PI / 4 + LAT0 / 2) ** N;
const f = (v: number) => (Number.isInteger(v) ? `${v}.0` : `${v}`);

const UNIFORMS = /* glsl */ `
uniform riskUniforms {
  float useWeather;
  float frame;
  float gate;
  float pHide;
  float pMax;
  float standing;
  float field;
  vec4 uv;
  vec4 grid;
  vec4 merc;
} risk;
`;

export const riskModule = {
  name: "risk",
  vs: /* glsl */ `${UNIFORMS}
out vec2 riskLonLat;
`,
  fs: /* glsl */ `${UNIFORMS}
precision highp sampler2DArray;
uniform sampler2D rampTex;
uniform sampler2D fuelRampTex;
uniform sampler2D fuelTex;
uniform sampler2D maskTex;
uniform sampler2DArray wxTex;
in vec2 riskLonLat;

/** A point (degrees) on the HRRR grid: column and row, whole numbers at pixel centers (Lambert conformal on a sphere). */
vec2 riskGrid(vec2 ll) {
  float rho = ${f(R * F)} / pow(tan(0.7853981634 + radians(ll.y) * 0.5), ${f(N)});
  float th = ${f(N)} * radians(ll.x + 97.5);
  return vec2((rho * sin(th) - (${f(HRRR.x0)})) / 3000.0 - 0.5, (${f(HRRR.y0)} - (${f(RHO0)} - rho * cos(th))) / 3000.0 - 0.5);
}

/**
 * The weather score (logits) at a grid point and frame, read as a smooth surface (a cubic B-spline over the 4 x 4 pixels
 * around the point, as four blended reads) and blended between the two frames around the time. Pixels not in hand
 * drop out by the mask. -99: no weather here.
 */
float riskWx(vec2 g, float frame) {
  vec2 size = risk.grid.zw;
  vec2 st = g - risk.grid.xy;
  if (st.x < -0.5 || st.y < -0.5 || st.x > size.x - 0.5 || st.y > size.y - 0.5) return -99.0;
  vec2 i = floor(st);
  vec2 t = st - i;
  vec2 t2 = t * t, t3 = t2 * t;
  vec2 w0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
  vec2 w1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
  vec2 w3 = t3 / 6.0;
  vec2 w2 = 1.0 - w0 - w1 - w3;
  vec2 g0 = w0 + w1, g1 = w2 + w3;
  vec2 p0 = (i - 0.5 + w1 / g0) / size;
  vec2 p1 = (i + 1.5 + w3 / g1) / size;
  float l0 = floor(frame), fr = frame - l0;
  float acc = 0.0, m = 0.0;
  for (int k = 0; k < 4; k++) {
    vec2 uv = vec2(k % 2 == 0 ? p0.x : p1.x, k < 2 ? p0.y : p1.y);
    float w = (k % 2 == 0 ? g0.x : g1.x) * (k < 2 ? g0.y : g1.y);
    float a = texture(wxTex, vec3(uv, l0)).r;
    if (fr > 0.0) a = mix(a, texture(wxTex, vec3(uv, l0 + 1.0)).r, fr);
    acc += w * a;
    m += w * texture(maskTex, uv).r;
  }
  return m > 0.25 ? ${f(TERM_LO)} + (acc / m * 255.0 - 1.0) / 254.0 * ${f(TERM_HI - TERM_LO)} : -99.0;
}

/** The color of a place: its fuel (0..1), how much of the pixel is ground (0..1), and where it is (degrees). */
vec4 riskShade(float fuel, float cover, vec2 lonlat) {
  if (risk.useWeather < 0.5) {
    if (cover < 0.01) discard;
    return vec4(texture(fuelRampTex, vec2(fuel, 0.5)).rgb, cover);
  }
  float wx = riskWx(riskGrid(lonlat), risk.frame);
  bool has = wx > -90.0;
  // the ground's score, and with the weather on top where the weather is across the gate (over half a logit)
  float g = mix(${f(GROUND_LO)}, ${f(GROUND_HI)}, fuel);
  float lit = has ? smoothstep(risk.gate, risk.gate + 0.5, wx) : 0.0;
  float z = mix(g, g + wx, lit);
  float t = clamp((1.0 / (1.0 + exp(-z)) - risk.pHide) / max(risk.pMax - risk.pHide, 0.0001), 0.0, 1.0);
  vec3 rgb = texture(rampTex, vec2(t, 0.5)).rgb;
  // the standing picture in gray (the ramp's lightness), color only where the weather lifts the pixel
  rgb = mix(vec3(dot(rgb, vec3(0.2126, 0.7152, 0.0722))), rgb, lit);
  float a = cover * mix(risk.standing, 1.0, lit);
  if (has && risk.field > 0.0) {
    // the weather under the ground: dark three logits below the gate, pale one above it
    float tf = clamp((wx - (risk.gate - 3.0)) / 4.0, 0.0, 1.0);
    vec3 under = vec3(0.10 + 0.80 * tf);
    float ao = a + risk.field * (1.0 - a);
    rgb = (rgb * a + under * risk.field * (1.0 - a)) / max(ao, 1e-4);
    a = ao;
  }
  if (a < 0.01) discard;
  return vec4(rgb, a);
}
`,
  uniformTypes: { useWeather: "f32", frame: "f32", gate: "f32", pHide: "f32", pMax: "f32", standing: "f32", field: "f32", uv: "vec4<f32>", grid: "vec4<f32>", merc: "vec4<f32>" },
} as const satisfies ShaderModule;

/** A tile: the fuel texture read at the fragment (uv: the part of the texture this square shows), the place from the square's Mercator box. */
export const TILE_INJECT = {
  "fs:DECKGL_FILTER_COLOR": /* glsl */ `
    vec4 s = texture(fuelTex, risk.uv.xy + geometry.uv * risk.uv.zw);
    vec2 m = mix(risk.merc.xy, risk.merc.zw, geometry.uv);
    vec2 ll = vec2(m.x * 360.0 - 180.0, degrees(atan(sinh(3.14159265359 * (1.0 - 2.0 * m.y)))));
    color = riskShade(s.a > 0.003 ? s.r / s.a : 0.0, s.a, ll);
  `,
};

/** A cell: its fuel is the red of its fill color, its place its own corners. */
export const CELL_INJECT = {
  "vs:#main-end": /* glsl */ `
    riskLonLat = geometry.worldPosition.xy;
  `,
  "fs:DECKGL_FILTER_COLOR": /* glsl */ `
    color = riskShade(color.r, 1.0, riskLonLat);
  `,
};
