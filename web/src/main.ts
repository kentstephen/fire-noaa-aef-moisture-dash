import "maplibre-gl/dist/maplibre-gl.css";
import "./page.css";

import { MapLibreOverlay } from "@deck.gl/maplibre";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

import { cellM, createCells } from "./cells";
import { createFine } from "./fine";
import { createOverture } from "./overture";
import { cssGradient, VIRIDIS_50, volT } from "./ramp";

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
    <div class="zoom"></div>
    <div class="status ground"></div>
    <div class="status"></div>
    <div class="keys"><kbd>F</kbd> full screen &nbsp; <kbd>H</kbd> all of California</div>
    <div class="credits">AlphaEarth Foundations embeddings (Google, via Source Cooperative) &middot; LANDFIRE-trained volume head &middot; A5 cells &middot; basemap &copy; CARTO, &copy; OpenStreetMap contributors &middot; Overture Maps Foundation</div>
  </div>`;
top.append(card);
document.body.append(bar, tag, top, tip);
const $ = <T extends HTMLElement = HTMLElement>(q: string) => card.querySelector<T>(q)!;
$(".at-ramp").style.background = $(".bar").style.background = cssGradient(VIRIDIS_50);
$(".at-hd").onclick = () => card.classList.toggle("collapsed");

const map = new maplibregl.Map({ container: mapEl, style: STYLE, bounds: HOME, fitBoundsOptions: { padding: 24 }, attributionControl: false, hash: true, ...FLAT });
const overlay = new MapLibreOverlay({ interleaved: true, layers: [], onError: (e) => say(`deck: ${e?.message ?? e}`) });
map.addControl(overlay as unknown as maplibregl.IControl);
(globalThis as unknown as { __map?: maplibregl.Map }).__map = map;
// for the headless checks: the two cell stores
queueMicrotask(() => Object.assign(globalThis, { __fuel: { cells, fine, ground } }));

let note = "";
function say(m: string) {
  note = m;
  console.warn(m);
  status();
}

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
  overlay.setProps({ layers: [...cells.layers(firstLabel), ...fine.layers(firstLabel), ...ground.layers(firstLabel)] });
  map.triggerRepaint();
}

const km = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(m >= 10_000 ? 0 : 1)} km` : m >= 100 ? `${Math.round(m / 10) * 10} m` : `${Math.round(m)} m`);
function status() {
  const p = cells.progress(), f = fine.progress();
  const loading = !p.ready || !p.done || !f.done;
  const got = p.got + f.got, total = p.total + f.total;
  bar.classList.toggle("on", loading);
  (bar.firstElementChild as HTMLElement).style.width = `${total ? (100 * got) / total : 0}%`;
  tag.classList.toggle("on", loading);
  tag.innerHTML = !p.ready
    ? "Opening the Parquet files"
    : !p.done
      ? `Reading cells<span>${p.got} of ${p.total} row groups</span>`
      : `Reading the 10 m embeddings<span>${f.got} of ${f.total} chunks</span>`;
  $(".zoom").textContent = fine.active
    ? `Cells about ${km(cellM(fine.res))} across (A5 resolution ${fine.res}), made here from the 10 m embeddings`
    : cells.res
      ? `Cells about ${km(cellM(cells.res))} across (A5 resolution ${cells.res})`
      : "";
  const g = ground.status();
  $(".ground").textContent = !ground.active
    ? ""
    : g.reading.length
      ? `Reading ${g.reading.join(", ")} from Overture Maps`
      : `Roads thin, power lines bold, buildings from zoom 14 (Overture Maps)${g.notes.length ? `; ${g.notes.join("; ")}` : ""}`;
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
function readout() {
  const c = pointer && (fine.at(pointer.lon, pointer.lat) ?? cells.at(pointer.lon, pointer.lat));
  tip.classList.toggle("on", !!c);
  if (!c || !pointer) return;
  tip.innerHTML = `<div class="v"><b>${num(c.vol)}</b> m</div><div class="s">cover times height</div><div class="s">cell about ${km(cellM(c.res))} across</div>${c.res >= 19 ? "" : `<div class="s">${c.n.toLocaleString("en-US")} pixels of 10 m</div>`}`;
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
});

map.once("load", () => {
  firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
  askView();
});
// deck adds its slot under the labels only once the style reports loaded; set the layers again when the map settles
map.on("idle", () => {
  if (firstLabel && !map.getLayer(`deck-maplibre-layer-group-before:${firstLabel}`) && cells.progress().got) draw();
});
status();
