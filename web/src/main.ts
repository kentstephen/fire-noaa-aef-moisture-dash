import "maplibre-gl/dist/maplibre-gl.css";
import "./page.css";

import { MapLibreOverlay } from "@deck.gl/maplibre";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

import { cellM, createCells } from "./cells";
import { createFine } from "./fine";
import { createOverture } from "./overture";
import { cssGradient, VIRIDIS_50, volT } from "./ramp";
import { createWeather, LAGS } from "./wx";
import { hrrrXY } from "./wx.grid";

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
  <div class="at-hd"><span class="t">Fuel</span><i class="at-ramp"></i><button class="at-cb" aria-label="Fold the card">${CHEV}</button></div>
  <div class="body">
    <div class="sub">Vegetation volume across California, 2025</div>
    <div class="key">
      <div class="bar"></div>
      <div class="ticks">${TICKS.map((v) => `<span style="left:${(100 * volT(v)).toFixed(1)}%">${v}</span>`).join("")}</div>
      <div class="ends"><span>least fuel</span><span>most fuel</span></div>
      <div class="unit">Canopy cover times height, in meters, averaged over each cell</div>
    </div>
    <div class="wx">
      <div class="seg-f onoff"><button data-on="1">Primed fuel</button><button data-on="0">All fuel</button></div>
      <div class="wxbody">
        <div class="dayrow"><button class="step" data-d="-1" aria-label="The day before">&lsaquo;</button><span class="day"></span><button class="step" data-d="1" aria-label="The day after">&rsaquo;</button><button class="step play" aria-label="Play the days">&#9654;</button></div>
        <div class="wxkey"><span><i class="c"></i>primed by the weather</span><span><i class="g"></i>not primed</span></div>
        <div class="at-row"><span class="at-lab">Pace</span><span class="seg-f lag">${LAGS.map((l, i) => `<button data-lag="${i}">${l} hour</button>`).join("")}</span></div>
        <div class="at-row"><span class="at-lab">Primed at</span><input type="range" class="line" min="2" max="60" step="1" aria-label="Fosberg at which fuel counts as primed"><span class="linev"></span></div>
        <div class="sub">Fosberg fire weather, built up over days of dry air and wind and let go by rain and damp air</div>
      </div>
    </div>
    <div class="zoom"></div>
    <div class="status ground"></div>
    <div class="status"></div>
    <div class="keys"><kbd>F</kbd> full screen &nbsp; <kbd>H</kbd> all of California &nbsp; <kbd>W</kbd> weather &nbsp; <kbd>[</kbd><kbd>]</kbd> day</div>
    <div class="credits">AlphaEarth Foundations embeddings (Google, via Source Cooperative) &middot; LANDFIRE-trained volume head &middot; A5 cells &middot; basemap &copy; CARTO, &copy; OpenStreetMap contributors &middot; Overture Maps Foundation</div>
  </div>`;
top.append(card);
document.body.append(bar, tag, top, tip);
const $ = <T extends HTMLElement = HTMLElement>(q: string) => card.querySelector<T>(q)!;
$(".at-ramp").style.background = $(".bar").style.background = cssGradient(VIRIDIS_50);
$(".at-hd").onclick = () => card.classList.toggle("collapsed");
$(".wxkey .c").style.background = cssGradient(VIRIDIS_50);
$(".wxkey .g").style.background = "linear-gradient(90deg, #060606, #f4f4f4)";

const map = new maplibregl.Map({ container: mapEl, style: STYLE, bounds: HOME, fitBoundsOptions: { padding: 24 }, attributionControl: false, hash: true, ...FLAT });
const overlay = new MapLibreOverlay({ interleaved: true, layers: [], onError: (e) => say(`deck: ${e?.message ?? e}`) });
map.addControl(overlay as unknown as maplibregl.IControl);
(globalThis as unknown as { __map?: maplibregl.Map }).__map = map;
// for the headless checks: the two cell stores
queueMicrotask(() => Object.assign(globalThis, { __fuel: { cells, fine, ground, wx }, __px: (lon: number, lat: number) => (hrrrXY(lon, lat, tipXY, 0), [tipXY[0], tipXY[1]]) }));

let note = "";
function say(m: string) {
  note = m;
  console.warn(m);
  status();
}

// the weather: which fuel is primed
const wx = createWeather(() => {
  draw();
  status();
  readout();
  wxPanel();
}, say);

// the cells draw under the basemap's labels
let firstLabel: string | undefined;
const cells = createCells(() => {
  draw();
  status();
  readout();
}, say);
// zoomed in, finer cells made here from the 10 m embeddings, over the baked ones
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
  cells.paint(wx);
  fine.paint(wx);
  wx.need(cells.blocks());
  overlay.setProps({ layers: [...cells.layers(firstLabel), ...fine.layers(firstLabel), ...ground.layers(firstLabel)] });
  map.triggerRepaint();
}

const km = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(m >= 10_000 ? 0 : 1)} km` : m >= 100 ? `${Math.round(m / 10) * 10} m` : `${Math.round(m)} m`);
function status() {
  const p = cells.progress(), f = fine.progress(), g = ground.status(), wp = wx.progress();
  const loading = !p.ready || !p.done || !f.done || g.reading.length > 0 || !wp.done;
  const got = p.got + f.got + wp.got, total = p.total + f.total + wp.total;
  bar.classList.toggle("on", loading);
  (bar.firstElementChild as HTMLElement).style.width = `${total ? (100 * got) / total : 0}%`;
  tag.classList.toggle("on", loading);
  tag.innerHTML = !p.ready
    ? "Opening the Parquet files"
    : !p.done
      ? `Reading cells<span>${p.got} of ${p.total} row groups</span>`
      : !f.done
        ? `Reading the 10 m embeddings<span>${f.got} of ${f.total} chunks</span>`
        : !wp.done
          ? `Reading the weather<span>${wp.got} of ${wp.total || "?"} blocks</span>`
          : `Reading ${g.reading.join(", ")}<span>Overture Maps</span>`;
  $(".zoom").textContent = fine.active
    ? `Cells about ${km(cellM(fine.res))} across (A5 resolution ${fine.res}), made here from the 10 m embeddings`
    : cells.res
      ? `Cells about ${km(cellM(cells.res))} across (A5 resolution ${cells.res})`
      : "";
  $(".ground").textContent = !ground.active
    ? ""
    : g.reading.length
      ? `Reading ${g.reading.join(", ")} from Overture Maps`
      : `Roads thin, power lines bold, building outlines from zoom 14 (Overture Maps)${g.notes.length ? `; ${g.notes.join("; ")}` : ""}`;
  $(".status:not(.ground)").textContent = note;
}

let viewTimer = 0;
function askView() {
  const b = map.getBounds();
  const box: [number, number, number, number] = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  cells.view(map.getZoom(), box);
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

// the cell under the pointer
let pointer: { x: number; y: number; lon: number; lat: number } | null = null;
const num = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));
const tipXY = new Float32Array(2);
function readout() {
  const c = pointer && (fine.at(pointer.lon, pointer.lat) ?? cells.at(pointer.lon, pointer.lat));
  tip.classList.toggle("on", !!c);
  if (!c || !pointer) return;
  tip.innerHTML = `<div class="v"><b>${num(c.vol)}</b> m</div><div class="s">cover times height</div><div class="s">cell about ${km(cellM(c.res))} across</div>${c.res >= 19 ? "" : `<div class="s">${c.n.toLocaleString("en-US")} pixels of 10 m</div>`}`;
  if (wx.st.on) {
    hrrrXY(pointer.lon, pointer.lat, tipXY, 0);
    const v = wx.value(tipXY[0]!, tipXY[1]!);
    tip.innerHTML += v === v ? `<div class="s">Fosberg, built up: ${v.toFixed(0)}${v >= wx.st.line ? ", primed" : ""}</div>` : `<div class="s">weather not read yet</div>`;
  }
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

// the weather's controls
const dayName = (ms: number) => new Date(ms).toLocaleDateString("en-US", { timeZone: "America/Los_Angeles", weekday: "short", month: "short", day: "numeric" });
function wxPanel() {
  card.classList.toggle("nowx", !wx.st.on);
  for (const b of card.querySelectorAll<HTMLElement>(".onoff button")) b.classList.toggle("on", (b.dataset.on === "1") === wx.st.on);
  for (const b of card.querySelectorAll<HTMLElement>(".lag button")) b.classList.toggle("on", Number(b.dataset.lag) === wx.st.lag);
  $(".day").textContent = wx.days.length && wx.st.day >= 0 ? dayName(wx.days[wx.st.day]!) : "reading the weather";
  $<HTMLInputElement>(".line").value = String(wx.st.line);
  $(".linev").textContent = String(wx.st.line);
  $(".play").innerHTML = playing ? "&#10073;&#10073;" : "&#9654;";
}
let playing = 0;
const stepDay = (d: number) => wx.days.length && wx.set({ day: (wx.st.day + d + wx.days.length) % wx.days.length });
function togglePlay() {
  if (playing) clearInterval(playing);
  playing = playing ? 0 : window.setInterval(() => stepDay(1), 700);
  wxPanel();
}
for (const b of card.querySelectorAll<HTMLElement>(".onoff button")) b.onclick = () => wx.set({ on: b.dataset.on === "1" });
for (const b of card.querySelectorAll<HTMLElement>(".lag button")) b.onclick = () => wx.set({ lag: Number(b.dataset.lag) });
for (const b of card.querySelectorAll<HTMLElement>(".step[data-d]")) b.onclick = () => stepDay(Number(b.dataset.d));
$(".play").onclick = togglePlay;
$<HTMLInputElement>(".line").oninput = (e) => wx.set({ line: Number((e.target as HTMLInputElement).value) });

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
  else if (e.key === "[") stepDay(-1);
  else if (e.key === "]") stepDay(1);
});

map.once("load", () => {
  firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
  askView();
});
// deck adds its slot under the labels only once the style reports loaded; set the layers again when the map settles
map.on("idle", () => {
  if (firstLabel && !map.getLayer(`deck-maplibre-layer-group-before:${firstLabel}`) && cells.progress().got) draw();
});
wx.set({});
status();
