#!/usr/bin/env python3
"""
Check google-scrape → public.places → tat.places mapping.
Find which public.places IDs are NOT yet in tat.places and prepare migration.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymongo import MongoClient
import psycopg2
from config import PG_CONFIG

conn = psycopg2.connect(**PG_CONFIG)
cur = conn.cursor()

# Get all public.places not in tat.places
cur.execute("""
    SELECT p.id, p.name, p.name_en, p.province_id, p.latitude, p.longitude,
           p.rating, p.thumbnail_url, p.detail, p.detail_en
    FROM public.places p
    WHERE p.id NOT IN (SELECT place_id FROM tat.places)
    LIMIT 5
""")
cols = [d[0] for d in cur.description]
print("=== public.places NOT in tat.places (sample 5) ===")
for row in cur.fetchall():
    for c, v in zip(cols, row):
        val = str(v)[:120] if v else "NULL"
        print(f"  {c}: {val}")
    print()

# Count
cur.execute("""SELECT count(*) FROM public.places WHERE id NOT IN (SELECT place_id FROM tat.places)""")
missing = cur.fetchone()[0]
print(f"Total public.places NOT in tat.places: {missing}")

# Check province_id overlap
cur.execute("""
    SELECT DISTINCT p.province_id 
    FROM public.places p 
    WHERE p.id NOT IN (SELECT place_id FROM tat.places)
    AND p.province_id IS NOT NULL
    ORDER BY p.province_id
""")
prov_ids = [r[0] for r in cur.fetchall()]
print(f"\nDistinct province_ids in new places: {len(prov_ids)}")
print(f"Range: {min(prov_ids)}-{max(prov_ids)}")

# Check if province_ids match tat.provinces
cur.execute("SELECT province_id FROM tat.provinces")
tat_provs = set(r[0] for r in cur.fetchall())
new_provs_not_in_tat = set(prov_ids) - tat_provs
print(f"Province IDs not in tat.provinces: {new_provs_not_in_tat}")

# Check tat.places max place_id 
cur.execute("SELECT max(place_id) FROM tat.places")
max_pid = cur.fetchone()[0]
print(f"\ntat.places max place_id: {max_pid}")

# Get tat.categories for mapping
cur.execute("SELECT category_id, name FROM tat.categories ORDER BY category_id")
cats = cur.fetchall()
print("\n=== tat.categories ===")
for c in cats:
    print(f"  {c[0]}: {c[1]}")

# Check what best_season values exist
cur.execute("SELECT DISTINCT best_season FROM public.places")
seasons = [r[0] for r in cur.fetchall()]
print(f"\nbest_season values: {seasons}")

conn.close()

# Google-scrape: check how titles map to public.places
local = MongoClient("mongodb://localhost:27018", serverSelectionTimeoutMS=5000)
gdb = local["google-scrape"]

# Sample from each collection
print("\n=== Google-scrape samples for category mapping ===")
for col_name in sorted(gdb.list_collection_names()):
    doc = gdb[col_name].find_one()
    if doc:
        title = doc.get("title", "")
        cat = doc.get("category", "")
        addr = str(doc.get("address", ""))[:80]
        desc = str(doc.get("descriptions") or doc.get("description") or "")[:80]
        rating = doc.get("review_rating")
        reviews = doc.get("review_count")
        print(f"  {col_name}: title={title}, cat={cat}, rating={rating}, reviews={reviews}")
        print(f"    addr: {addr}")
        print(f"    desc: {desc}")

local.close()
print("\nDONE")
