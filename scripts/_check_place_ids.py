#!/usr/bin/env python3
"""Check place_id overlap between content_metadata public and existing tables."""
from pymongo import MongoClient
import psycopg2

local = MongoClient("mongodb://localhost:27018", serverSelectionTimeoutMS=5000)
db = local["tiktok_scraper"]

# Get distinct place_ids from public content_metadata
print("Getting distinct place_ids from content_metadata (source=public)...")
public_place_ids = set(db.content_metadata.distinct("place_id", {"source": "public"}))
print(f"  Distinct place_ids in public content_metadata: {len(public_place_ids)}")

# Get distinct place_ids from place_videos  
pv_place_ids = set(db.place_videos.distinct("place_id"))
print(f"  Distinct place_ids in place_videos: {len(pv_place_ids)}")

# Overlap
overlap = public_place_ids & pv_place_ids
only_public = public_place_ids - pv_place_ids
only_pv = pv_place_ids - public_place_ids
print(f"  Overlap: {len(overlap)}")
print(f"  Only in public CM: {len(only_public)}")
print(f"  Only in place_videos: {len(only_pv)}")

# Check ranges
if public_place_ids:
    print(f"  Public place_id range: {min(public_place_ids)}-{max(public_place_ids)}")
if pv_place_ids:
    print(f"  place_videos place_id range: {min(pv_place_ids)}-{max(pv_place_ids)}")

local.close()

# Check against PostgreSQL
print("\n=== PostgreSQL place_id ranges ===")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PG_CONFIG
conn = psycopg2.connect(**PG_CONFIG)
cur = conn.cursor()

# tat.places
cur.execute("SELECT count(*), min(place_id), max(place_id) FROM tat.places")
r = cur.fetchone()
print(f"  tat.places: {r[0]} rows, place_id {r[1]}-{r[2]}")

# public.places
cur.execute("SELECT count(*), min(id), max(id) FROM public.places")
r = cur.fetchone()
print(f"  public.places: {r[0]} rows, id {r[1]}-{r[2]}")

# Check how many public CM place_ids exist in public.places
cur.execute("SELECT id FROM public.places")
public_pg_ids = set(r[0] for r in cur.fetchall())
in_public_pg = public_place_ids & public_pg_ids
print(f"\n  public CM place_ids found in public.places: {len(in_public_pg)}/{len(public_place_ids)}")

# Check how many exist in tat.places
cur.execute("SELECT place_id FROM tat.places")
tat_pg_ids = set(r[0] for r in cur.fetchall())
in_tat = public_place_ids & tat_pg_ids
print(f"  public CM place_ids found in tat.places: {len(in_tat)}/{len(public_place_ids)}")

# Check place_videos place_ids against both tables
pv_in_tat = pv_place_ids & tat_pg_ids
pv_in_public = pv_place_ids & public_pg_ids
print(f"\n  place_videos place_ids in tat.places: {len(pv_in_tat)}/{len(pv_place_ids)}")
print(f"  place_videos place_ids in public.places: {len(pv_in_public)}/{len(pv_place_ids)}")

conn.close()
print("\nDONE")
