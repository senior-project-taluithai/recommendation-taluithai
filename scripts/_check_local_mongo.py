#!/usr/bin/env python3
"""Quick check of local MongoDB and PostgreSQL data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymongo import MongoClient
import psycopg2

# ── MongoDB ──
c = MongoClient("mongodb://localhost:27018", serverSelectionTimeoutMS=5000)
db = c["tiktok_scraper"]

print("=== MongoDB content_metadata ===")
total = db.content_metadata.estimated_document_count()
print(f"  total: {total}")

# Count public vs old (using indexed field lookup)
public_sample = list(db.content_metadata.find({"source": "public"}).limit(1))
old_sample = list(db.content_metadata.find({"source": {"$exists": False}}).limit(1))
print(f"  has public entries: {len(public_sample) > 0}")
print(f"  has old entries: {len(old_sample) > 0}")

# Place_videos  
pv = db.place_videos.estimated_document_count()
print(f"\n  place_videos docs: {pv}")

# Schema comparison
if public_sample:
    m = public_sample[0].get("metadata", {})
    print(f"\n  PUBLIC schema: desc={bool(m.get('desc'))}, "
          f"challenges={bool(m.get('challenges'))}, "
          f"stats={bool(m.get('stats'))}, "
          f"author.id={m.get('author',{}).get('id')}, "
          f"stickersOnItem={bool(m.get('stickersOnItem'))}, "
          f"poi={bool(m.get('poi'))}")
    print(f"  Top-level: place_id={public_sample[0].get('place_id')}, "
          f"place_name={public_sample[0].get('place_name')}")

if old_sample:
    vm = old_sample[0].get("metadata", {}).get("video_metadata", {})
    print(f"\n  OLD schema: description={bool(vm.get('description'))}, "
          f"hashtags={bool(vm.get('hashtags'))}, "
          f"playcount={vm.get('playcount')}, "
          f"author_id={vm.get('author_id')}, "
          f"stickers_on_item={bool(vm.get('stickers_on_item'))}")

c.close()

# ── PostgreSQL ──
print("\n=== PostgreSQL ===")
from config import PG_CONFIG
conn = psycopg2.connect(**PG_CONFIG)
cur = conn.cursor()

# List schemas
cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast')")
schemas = [r[0] for r in cur.fetchall()]
print(f"  schemas: {schemas}")

# Count places in each schema
for schema in schemas:
    try:
        cur.execute(f"SELECT count(*) FROM {schema}.places")
        cnt = cur.fetchone()[0]
        print(f"  {schema}.places: {cnt}")
    except Exception as e:
        conn.rollback()

# Check public.places columns
try:
    cur.execute("""
        SELECT column_name, data_type FROM information_schema.columns 
        WHERE table_schema='public' AND table_name='places' 
        ORDER BY ordinal_position
    """)
    cols = cur.fetchall()
    if cols:
        print(f"\n  public.places columns ({len(cols)}):")
        for name, dtype in cols[:20]:
            print(f"    {name}: {dtype}")
except Exception as e:
    conn.rollback()
    print(f"  public.places: {e}")

conn.close()
print("\nDONE")
