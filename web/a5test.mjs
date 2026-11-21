import * as a5 from "a5-js";
console.log(Object.keys(a5).join(" "));
const c = a5.lonLatToCell([-120, 37], 15);
console.log("js", c.toString(), "duckdb 1937941052535078912", c === 1937941052535078912n);
const p = a5.cellToParent(c, 12);
const kids = a5.cellToChildren(p, 15);
const sorted = [...kids].sort((a, b) => (a < b ? -1 : 1));
console.log("parent", p.toString(), "kids", kids.length, "already sorted", kids.every((k, i) => k === sorted[i]), "span", (sorted.at(-1) - sorted[0]).toString(), "step", (sorted[1] - sorted[0]).toString());
let t = performance.now(); let n = 0;
for (let i = 0; i < 20000; i++) { n += a5.cellToBoundary(kids[i % kids.length]).length; }
console.log("boundary us/cell", ((performance.now() - t) / 20000 * 1000).toFixed(1), "verts", n / 20000, JSON.stringify(a5.cellToBoundary(c)));
t = performance.now();
for (let i = 0; i < 20000; i++) a5.lonLatToCell([-120 + i * 1e-4, 37], 12);
console.log("lonLatToCell us", ((performance.now() - t) / 20000 * 1000).toFixed(1));
console.log(JSON.stringify(a5.cellToLonLat(c)));
