# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=2.0.1"]
# ///
"""The baked cells as Parquet the browser reads directly: one file per A5 resolution, inside California.

data/parts/*.parquet (bake_fuel.py) -> web/public/data/fuel_r<R>.parquet for R = --lo .. 15, each with
  cell UBIGINT   the A5 cell
  vol  FLOAT     mean vegetation volume of its 10 m pixels, m (cover x height)
  n    INTEGER   10 m pixels in the cell
sorted by cell, in small row groups. A5 IDs follow the hierarchy, so a place is a few runs of IDs and the browser
range-reads only the row groups that hold them. Which row groups hold which IDs is in fuel_index.parquet
(res, rg, lo, hi), because the browser's Parquet reader does not expose row group statistics.

A cell is in California when its center is inside the Census 1:500k outline (data/ca_500k.parquet, WKB).
Coarser cells are the pixel-weighted mean of the resolution 15 cells kept.

RAM: the 13.7 M resolution 15 rows and their sort, about 3 GB at the peak; DuckDB is capped at 8 GB.

    uv run scripts/levels.py [--lo 8]
"""
import argparse
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "public" / "data"
ROWS = 32768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET allow_extension_repositories = 'allowed'")
    con.execute("LOAD a5 FROM query_farm")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    t = time.time()
    con.execute(f"""CREATE TABLE raw AS SELECT cell, sum(vol)::DOUBLE AS vol, sum(n)::BIGINT AS n
                    FROM '{ROOT}/data/parts/*.parquet' GROUP BY 1""")
    print(f"merged {con.execute('SELECT count(*) FROM raw').fetchone()[0]:,} cells  {time.time() - t:.0f} s", flush=True)
    t = time.time()
    # the outline is tested once per resolution 11 cell (about 3 km); only cells of the ones it crosses are tested one by one
    con.execute(f"CREATE TABLE ca AS SELECT ST_GeomFromWKB(wkb) AS g FROM '{ROOT}/data/ca_500k.parquet'")
    con.execute("""CREATE TABLE big AS SELECT p, ST_Within(a5_cell_to_geometry(p), g) AS inside,
                       ST_Intersects(a5_cell_to_geometry(p), g) AS touches
                   FROM (SELECT DISTINCT a5_cell_to_parent(cell, 11) AS p FROM raw), ca""")
    con.execute("""CREATE TABLE r15 AS SELECT r.cell, r.vol, r.n FROM raw r
                   JOIN big b ON a5_cell_to_parent(r.cell, 11) = b.p, ca
                   WHERE b.inside OR (b.touches AND ST_Contains(g, a5_cell_to_point(r.cell)))""")
    print(f"in California {con.execute('SELECT count(*) FROM r15').fetchone()[0]:,} cells  {time.time() - t:.0f} s", flush=True)
    con.execute("SET threads = 1")  # one writer, so row groups come out in ID order
    index = []
    for res in range(15, a.lo - 1, -1):
        src = "r15" if res == 15 else f"(SELECT a5_cell_to_parent(cell, {res}) AS cell, sum(vol) AS vol, sum(n) AS n FROM r15 GROUP BY 1)"
        f = OUT / f"fuel_r{res}.parquet"
        con.execute(f"""COPY (SELECT cell, (vol / n)::FLOAT AS vol, n::INTEGER AS n FROM {src} ORDER BY cell)
                        TO '{f}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ROWS})""")
        rows = con.execute(f"""SELECT row_group_id, stats_min_value::UBIGINT, stats_max_value::UBIGINT FROM parquet_metadata('{f}')
                           WHERE path_in_schema = 'cell' ORDER BY 1""").fetchall()
        index += [(res, rg, lo, hi) for rg, lo, hi in rows]
        n = con.execute(f"SELECT count(*) FROM '{f}'").fetchone()[0]
        print(f"r{res}: {n:,} cells, {len(rows)} row groups, {f.stat().st_size / 1e6:.1f} MB", flush=True)
    con.execute("CREATE TABLE idx (res TINYINT, rg INTEGER, lo UBIGINT, hi UBIGINT)")
    con.executemany("INSERT INTO idx VALUES (?, ?, ?, ?)", index)
    con.execute(f"COPY (SELECT * FROM idx ORDER BY res, rg) TO '{OUT}/fuel_index.parquet' (FORMAT parquet, COMPRESSION zstd)")
    print(f"index: {len(index)} row groups", flush=True)


if __name__ == "__main__":
    main()
