#!/usr/bin/env python3
"""
Check cloud MongoDB and google-scrape schema for migration planning.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymongo import MongoClient
import psycopg2
from config import PG_CONFIG, MONGO_URI

# ── Cloud MongoDB ──
print("=== Cloud MongoDB ===")
try:
    cloud = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    dbs = cloud.list_database_names()
    print(f"  databases: {dbs}")
    for dbname in dbs:
        if dbname in ("admin", "config", "local"):
            continue
        d = cloud[dbname]
        cols = d.list_collection_names()
        for col in cols:
            cnt = d[col].estimated_document_count()
            print(f"  {dbname}.{col}: {cnt}")
    cloud.close()
except Exception as e:
    print(f"  ERROR: {e}")

# ── Local MongoDB google-scrape sample ──
print("\n=== Local google-scrape sample ===")
local = MongoClient("mongodb://localhost:27018", serverSelectionTimeoutMS=5000)
gdb = local["google-scrape"]
for col_name in gdb.list_collection_names():
    cnt = gdb[col_name].estimated_document_count()
    sample = gdb[col_name].find_one()
    keys = list(sample.keys()) if sample else []
    print(f"\n  {col_name}: {cnt} docs, keys={keys}")
    if sample:
        # Show key fields
        for k in ["name", "formatted_address", "rating", "user_ratings_total", 
                   "geometry", "types", "place_id", "vicinity", "lat", "lng",
                   "address", "province", "category"]:
            if k in sample:
                val = sample[k]
                if isinstance(val, str) and len(val) > 100:
                    val = val[:100] + "..."
                elif isinstance(val, dict):
                    val = {kk: vv for kk, vv in list(val.items())[:3]}
                print(f"    {k}: {val}")

local.close()

# ── PostgreSQL tat.places schema ──
print("\n=== tat.places full schema ===")
conn = psycopg2.connect(**PG_CONFIG)
cur = conn.cursor()
cur.execute("""
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns 
    WHERE table_schema='tat' AND table_name='places' 
    ORDER BY ordinal_position
""")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} (nullable={row[2]}, default={row[3]})")

# Also check public.places sample
cur.execute("SELECT * FROM public.places LIMIT 1")
cols = [d[0] for d in cur.description]
row = cur.fetchone()
print(f"\n=== public.places sample ===")
print(f"  columns: {cols}")
if row:
    for c, v in zip(cols, row):
        print(f"  {c}: {v}")

# Check max place_id in tat.places  
cur.execute("SELECT max(place_id), min(place_id) FROM tat.places")
r = cur.fetchone()
print(f"\n  tat.places place_id range: {r[0]} min={r[1]}")

# Check max id in public.places
cur.execute("SELECT max(id), min(id) FROM public.places")
r = cur.fetchone()
print(f"  public.places id range: max={r[0]} min={r[1]}")

# Check categories
cur.execute("SELECT * FROM tat.categories ORDER BY category_id")
print(f"\n=== tat.categories ===")
for row in cur.fetchall():
    print(f"  {row}")

conn.close()
print("\nDONE")
