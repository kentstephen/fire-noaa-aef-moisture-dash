import "maplibre-gl/dist/maplibre-gl.css";
import "./page.css";

import { MapLibreOverlay } from "@deck.gl/maplibre";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

import { cellM, createField } from "./field";
import { createFine } from "./fine";
import { createOverture } from "./overture";
import { cssGradient, INFERNO_50, VIRIDIS_50, volT } from "./ramp";
import { look } from "./risk";
import { createWeather } from "./wx";

maplibregl.setWorkerUrl(maplibreWorkerUrl);

const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
/** California, whole. */
const HOME: [number, number, number, number] = [-124.41, 32.53, -114.13, 42.01];
// the map never tilts
const FLAT = { maxPitch: 0, pitchWithRotate: false, touchPitch: false, dragRotate: false };
const CHEV = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9l6 6 6-6"/></svg>';

const mapEl = document.getElementById("map")!;

const bar = document.createElement("div");
bar.className = "ew-load";
bar.innerHTML = "<i></i>";
const tag = document.createElement("div");
tag.className = "ew-load-tag at-glass";
const tip = document.createElement("div");
tip.className = "tip at-glass";

const top = document.createElement("div");
top.className = "at-top";
const card = document.createElement("div");
card.className = "card at-glass";
const TICKS = [0, 0.1, 1, 10, 50];
card.innerHTML = `
  <div class="at-hd"><span class="t"></span><i class="at-ramp"></i><button class="at-cb" aria-label="Fold the card">${CHEV}</button></div>
  <div class="body">
    <div class="sub what"></div>
    <div class="key">
      <div class="bar"></div>
      <div class="ticks">${TICKS.map((v) => `<span style="left:${(100 * volT(v)).toFixed(1)}%">${v}</span>`).join("")}</div>
      <div class="ends"><span></span><span></span></div>
      <div class="unit"></div>
    </div>
    <div class="wx">
      <div class="seg-f onoff"><button data-on="1">Risk</button><button data-on="0">Fuel alone</button></div>
      <div class="wxbody">
        <div class="seg-f win"></div>
        <div class="dayrow"><button class="step play" aria-label="Play">&#9654;</button><span class="day"></span></div>
        <div class="at-row"><span class="at-lab">Time</span><input type="range" class="time" min="0" max="1" step="0.01" value="0" aria-label="Time within the window"></div>
        <label class="tick"><input type="checkbox" class="sus"> Sustained: ground the weather has lit stays lit, and lets go over a couple of days</label>
        <div class="wxkey"><span><i class="c"></i>lifted by the weather</span><span><i class="g"></i>not lifted</span></div>
        <div class="at-row"><span class="at-lab">Lights from</span><input type="range" class="line" min="-2" max="4" step="0.1" aria-label="Weather score at which the ground lights"><span class="linev"></span></div>
        <div class="sub">The gray under everything is the hour's fire weather (Fosberg, temperature, humidity, wind as one score): paler where hotter, drier, windier. Ground under it lights in color. <span class="span"></span></div>
      </div>
    </div>
    <div class="zoom"></div>
    <div class="status ground"></div>
    <div class="status"></div>
    <div class="keys"><kbd>F</kbd> full screen &nbsp; <kbd>H</kbd> all of California &nbsp; <kbd>W</kbd> weather &nbsp; <kbd>S</kbd> sustained &nbsp; <kbd>Space</kbd> play &nbsp; <kbd>[</kbd><kbd>]</kbd> step</div>
    <div class="credits">AlphaEarth Foundations embeddings (Google, via Source Cooperative) &middot; LANDFIRE-trained volume head &middot; A5 cells &middot; NOAA HRRR forecast (dynamical.org) &middot; basemap &copy; CARTO, &copy; OpenStreetMap contributors &middot; Overture Maps Foundation</div>
  </div>`;
top.append(card);
document.body.append(bar, tag, top, tip);
const $ = <T extends HTMLElement = HTMLElement>(q: string) => card.querySelector<T>(q)!;

$(".at-hd").onclick = () => card.classList.toggle("collapsed");
$(".wxkey .c").style.background = cssGradient(INFERNO_50);
$(".wxkey .g").style.background = "linear-gradient(90deg, #060606, #f4f4f4)";

const map = new maplibregl.Map({ container: mapEl, style: STYLE, bounds: HOME, fitBoundsOptions: { padding: 24 }, attributionControl: false, hash: true, ...FLAT });
const overlay = new MapLibreOverlay({ interleaved: true, layers: [], onError: (e) => say(`deck: ${e?.message ?? e}`) });
map.addControl(overlay as unknown as maplibregl.IControl);
(globalThis as unknown as { __map?: maplibregl.Map }).__map = map;
// for the headless checks: the two cell stores
queueMicrotask(() => Object.assign(globalThis, { __fuel: { field, fine, ground, wx, look } }));

let note = "";
function say(m: string) {
  note = m;
  console.warn(m);
  status();
}

// the weather: Fosberg hour by hour, read under the view
const wx = createWeather(() => {
  draw();
  status();
  readout();
  wxPanel();
}, say);

// the fuel as tiles made from the A5 cells; they draw under the basemap's labels
let firstLabel: string | undefined;
const field = createField(() => {
  draw();
  status();
  readout();
}, say);
// zoomed in, finer cells made here from the 10 m embeddings, over the tiles
const fine = createFine(() => {
  draw();
  status();
  readout();
}, say);

// zoomed in, what is on the ground as context: Overture's roads, power lines and buildings
const ground = createOverture(() => {
  draw();
  status();
}, say);

function draw() {
  const device = (overlay as unknown as { _deck?: { device?: Parameters<typeof wx.upload>[0] | null } })._deck?.device ?? undefined;
  if (device) wx.upload(device);
  look.weather = wx.st.on ? 1 : 0;
  overlay.setProps({ layers: [...field.layers(firstLabel, device), ...fine.layers(firstLabel), ...ground.layers(firstLabel)] });
  map.triggerRepaint();
}

const km = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(m >= 10_000 ? 0 : 1)} km` : m >= 100 ? `${Math.round(m / 10) * 10} m` : `${Math.round(m)} m`);
function status() {
  const p = field.progress(), f = fine.progress(), g = ground.status(), wp = wx.progress();
  const loading = !p.ready || !p.done || !f.done || g.reading.length > 0 || !wp.done;
  const got = p.got + f.got + wp.got, total = p.total + f.total + wp.total;
  bar.classList.toggle("on", loading);
  (bar.firstElementChild as HTMLElement).style.width = `${total ? (100 * got) / total : 0}%`;
  tag.classList.toggle("on", loading);
  tag.innerHTML = !p.ready
    ? "Opening the Parquet files"
    : !p.done
      ? `Reading the fuel<span>${p.got} of ${p.total} tiles</span>`
      : !f.done
        ? `Reading the 10 m embeddings<span>${f.got} of ${f.total} chunks</span>`
        : !wp.done
          ? `Reading the weather<span>${Math.floor(wp.got)} of ${wp.total || "?"} runs</span>`
          : `Reading ${g.reading.join(", ")}<span>Overture Maps</span>`;
  $(".zoom").textContent = fine.active
    ? `Cells about ${km(cellM(fine.res))} across (A5 resolution ${fine.res}), made here from the 10 m embeddings`
    : field.res
      ? `Fuel from A5 cells about ${km(cellM(field.res))} across (resolution ${field.res})`
      : "";
  $(".ground").textContent = !ground.active
    ? ""
    : g.reading.length
      ? `Reading ${g.reading.join(", ")} from Overture Maps`
      : `Roads and building outlines (from zoom 14) in white, power lines in cyan (Overture Maps)${g.notes.length ? `; ${g.notes.join("; ")}` : ""}`;
  $(".status:not(.ground)").textContent = note;
}

let viewTimer = 0;
function askView() {
  const b = map.getBounds();
  const box: [number, number, number, number] = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  field.view(map.getZoom(), box);
  wx.view(box);
  fine.view(map.getZoom(), box);
  ground.view(map.getZoom(), box);
  draw();
  status();
}
map.on("move", () => {
  if (viewTimer) return;
  viewTimer = window.setTimeout(() => {
    viewTimer = 0;
    askView();
  }, 150);
});
map.on("moveend", askView);

// what is under the pointer
let pointer: { x: number; y: number; lon: number; lat: number } | null = null;
const num = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));
/** The volume (m) at a place on the ramp's scale. */
const volOf = (t: number) => Math.max(0, Math.exp(Math.log(0.02) + t * (Math.log(50) - Math.log(0.02))) - 0.02);
function readout() {
  const fc = pointer && fine.at(pointer.lon, pointer.lat);
  const tc = pointer && !fc ? field.at(pointer.lon, pointer.lat) : null;
  const vol = fc ? fc.vol : tc ? volOf(tc.t) : NaN;
  const fos = pointer && wx.st.on ? wx.value(pointer.lon, pointer.lat, look.frame, look.sustained > 0.5) : NaN;
  const on = vol === vol || fos === fos;
  tip.classList.toggle("on", on);
  if (!on || !pointer) return;
  let html = vol === vol ? `<div class="v"><b>${num(vol)}</b> m of fuel</div><div class="s">canopy cover times height</div>` : `<div class="v">no ground here</div>`;
  if (fos === fos) {
    html += `<div class="v wxv"><b>${fos >= 0 ? "+" : "\u2212"}${Math.abs(fos).toFixed(1)}</b> weather${look.sustained > 0.5 ? ", sustained" : " this hour"}</div><div class="s">${fos >= look.gate ? "lifting the ground" : `lifts the ground from ${look.gate >= 0 ? "+" : "\u2212"}${Math.abs(look.gate).toFixed(1)}`}</div>`;
  } else if (wx.st.on) html += `<div class="s">weather not read yet</div>`;
  tip.innerHTML = html;
  const r = tip.getBoundingClientRect();
  const x = pointer.x + 16 + r.width > innerWidth ? pointer.x - 16 - r.width : pointer.x + 16;
  const y = Math.min(innerHeight - r.height - 8, pointer.y + 16);
  tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
}
map.on("mousemove", (e) => {
  pointer = { x: e.point.x, y: e.point.y, lon: e.lngLat.lng, lat: e.lngLat.lat };
  readout();
});
map.on("mouseout", () => {
  pointer = null;
  readout();
});

// the weather's controls: time moves smoothly between the frames, forward at RATE frames a second while playing
const when = (ms: number) => new Date(ms).toLocaleString("en-US", { timeZone: "America/New_York", weekday: "short", month: "short", day: "numeric", hour: "numeric", timeZoneName: "short" });
/**
 * The windows offered, four days each (the start of the last of two runs, UTC): days of 2026 when dry wind was over
 * much of California (most on August 3 and 4), and the present.
 */
const WINDOWS: { name: string; end: number | null }[] = [
  { name: "Aug 2 to 6", end: Date.UTC(2026, 7, 4, 12) },
  { name: "Now", end: null },
];
$(".win").innerHTML = WINDOWS.map((w, i) => `<button data-w="${i}">${w.name}</button>`).join("");
for (const b of card.querySelectorAll<HTMLElement>(".win button")) {
  b.onclick = () => {
    if (playing) togglePlay();
    moved = false;
    wx.set({ end: WINDOWS[Number(b.dataset.w)]!.end });
  };
}
/** Six hours of weather for each second of play. */
const RATE = 6;
let playing = 0, shownAt = 0, moved = false;
/** The frame nearest the present; the ones after it are forecast. */
const nowFrame = () => {
  const t = wx.times;
  if (!t.length) return 0;
  return Math.min(t.length - 1, Math.max(0, (Date.now() - t[0]!) / (t[1]! - t[0]!)));
};
const last = () => Math.max(0, wx.times.length - 1);
function setFrame(f: number) {
  look.frame = Math.min(last(), Math.max(0, f));
  moved = true;
  map.triggerRepaint();
}
function wxPanel() {
  const on = wx.st.on;
  card.classList.toggle("nowx", !on);
  $(".t").textContent = on ? "Fire risk" : "Fuel";
  $(".what").textContent = on ? "Fuel, lit where the fire weather lifts it, California" : "Vegetation volume across California, 2025";
  $(".at-ramp").style.background = $(".bar").style.background = cssGradient(on ? INFERNO_50 : VIRIDIS_50);
  $<HTMLInputElement>(".sus").checked = look.sustained > 0.5;
  card.querySelector<HTMLElement>(".key .ticks")!.style.display = on ? "none" : "";
  const ends = card.querySelectorAll<HTMLElement>(".key .ends span");
  ends[0]!.textContent = on ? "lower" : "least fuel";
  ends[1]!.textContent = on ? "higher: heavy fuel in hot, dry wind" : "most fuel";
  $(".unit").textContent = on ? "" : "Canopy cover times height, in meters";
  for (const b of card.querySelectorAll<HTMLElement>(".onoff button")) b.classList.toggle("on", (b.dataset.on === "1") === on);
  for (const b of card.querySelectorAll<HTMLElement>(".win button")) b.classList.toggle("on", WINDOWS[Number(b.dataset.w)]!.end === wx.st.end);
  $(".span").textContent = wx.st.end == null ? "Two days back and two ahead." : "Four days.";
  // the present until the time is moved
  if (!moved) look.frame = nowFrame();
  const t = $<HTMLInputElement>(".time");
  t.max = String(last());
  t.value = String(look.frame);
  const n = wx.times.length;
  const at = n ? wx.times[0]! + look.frame * (wx.times[1]! - wx.times[0]!) : 0;
  $(".day").textContent = n ? `${when(at)}${at > Date.now() + 1_800_000 ? ", forecast" : ""}` : "reading the weather";
  $<HTMLInputElement>(".line").value = String(look.gate);
  $(".linev").textContent = `${look.gate >= 0 ? "+" : "\u2212"}${Math.abs(look.gate).toFixed(1)}`;
  $(".play").innerHTML = playing ? "&#10073;&#10073;" : "&#9654;";
}
function tickPlay(now: number) {
  if (!playing) return;
  const dt = Math.min(0.1, (now - playing) / 1000);
  playing = now;
  // from the newest frame, play starts over at the oldest
  setFrame(look.frame >= last() ? 0 : look.frame + RATE * dt);
  if (now - shownAt > 200) {
    shownAt = now;
    wxPanel();
    readout();
  }
  if (look.frame >= last()) {
    playing = 0;
    wxPanel();
  } else requestAnimationFrame(tickPlay);
}
function togglePlay() {
  if (!wx.times.length) return;
  playing = playing ? 0 : performance.now();
  if (playing) requestAnimationFrame(tickPlay);
  wxPanel();
}
function step(d: number) {
  setFrame(Math.round(look.frame) + d);
  wxPanel();
  readout();
}
for (const b of card.querySelectorAll<HTMLElement>(".onoff button")) b.onclick = () => wx.set({ on: b.dataset.on === "1" });
$(".play").onclick = togglePlay;
$<HTMLInputElement>(".time").oninput = (e) => {
  setFrame(Number((e.target as HTMLInputElement).value));
  wxPanel();
  readout();
};
$<HTMLInputElement>(".sus").onchange = (e) => {
  look.sustained = (e.target as HTMLInputElement).checked ? 1 : 0;
  wxPanel();
  readout();
  map.triggerRepaint();
};
$<HTMLInputElement>(".line").oninput = (e) => {
  look.gate = Number((e.target as HTMLInputElement).value);
  wxPanel();
  map.triggerRepaint();
};

function toggleFull() {
  if (document.fullscreenElement) void document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => say("full screen is not allowed here"));
}
document.addEventListener("fullscreenchange", () => setTimeout(() => map.resize(), 30));
window.addEventListener("resize", () => map.resize());
window.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "f" || e.key === "F") toggleFull();
  else if (e.key === "h" || e.key === "H") map.fitBounds(HOME, { padding: 24 });
  else if (e.key === "w" || e.key === "W") wx.set({ on: !wx.st.on });
  else if (e.key === "s" || e.key === "S") $<HTMLInputElement>(".sus").click();
  else if (e.key === " ") (e.preventDefault(), togglePlay());
  else if (e.key === "[") step(-1);
  else if (e.key === "]") step(1);
});

map.once("load", () => {
  firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
  askView();
});
// deck adds its slot under the labels only once the style reports loaded; set the layers again when the map settles
map.on("idle", () => {
  if (firstLabel && !map.getLayer(`deck-maplibre-layer-group-before:${firstLabel}`) && field.progress().got) draw();
});
wx.set({ end: WINDOWS[0]!.end });
status();
