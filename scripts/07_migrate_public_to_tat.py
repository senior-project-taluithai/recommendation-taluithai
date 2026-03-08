#!/usr/bin/env python3
"""
Migrate public.places → tat.places
===================================
Insert new places from public.places (mainly from google-scrape)
into tat.places with proper category_id mapping.

public.places has: id, name, name_en, province_id, latitude, longitude,
                   best_season, rating, thumbnail_url, detail, detail_en

tat.places needs:  place_id, name, introduction, category_id, province_id,
                   latitude, longitude, address, google_avg_rating,
                   google_review_count, ...

Since public.places doesn't have category_id, we look up the corresponding
google-scrape doc to get the category, then map it to tat category_id.

Usage:
    python scripts/07_migrate_public_to_tat.py --dry-run
    python scripts/07_migrate_public_to_tat.py
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import psycopg2
from psycopg2.extras import execute_values
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from config import PG_CONFIG

# google-scrape collection → tat category_id
COLLECTION_CATEGORY_MAP = {
    "hotel": 2,        # ที่พัก
    "attraction": 3,   # สถานที่ท่องเที่ยว
    "musuem": 3,       # สถานที่ท่องเที่ยว (typo in original)
    "park": 3,         # สถานที่ท่องเที่ยว
    "temple": 3,       # สถานที่ท่องเที่ยว
    "cafe": 8,         # ร้านอาหาร กาแฟ เบเกอรี่
    "restaurant": 8,   # ร้านอาหาร กาแฟ เบเกอรี่
    "hospital": 13,    # สถานที่อื่นๆ
}

# google-scrape variants: some use "longitude" vs "longtitude"
VARIANT_A = {"attraction", "hotel"}  # use "longitude", "descriptions"


def build_google_scrape_index(mongo_uri: str) -> dict:
    """
    Build a lookup from place title → (category_id, description, address, rating, review_count)
    using google-scrape MongoDB data.
    Returns dict keyed by (title_normalized, lat_rounded, lng_rounded).
    """
    log.info("Building google-scrape index...")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    gdb = client["google-scrape"]

    index = {}  # (title, lat_round, lng_round) → info
    total = 0

    for col_name in gdb.list_collection_names():
        cat_id = COLLECTION_CATEGORY_MAP.get(col_name, 13)
        is_variant_a = col_name in VARIANT_A

        for doc in gdb[col_name].find():
            title = (doc.get("title") or "").strip()
            lat = doc.get("latitude")
            lng = doc.get("longitude") if is_variant_a else doc.get("longtitude")
            desc = doc.get("descriptions") if is_variant_a else doc.get("description")
            addr = doc.get("address") or ""
            rating = doc.get("review_rating")
            review_count = doc.get("review_count")
            google_place_id = doc.get("place_id")  # ChIJ... string
            phone = doc.get("phone")
            website = doc.get("website") if is_variant_a else doc.get("web_site")
            open_hours_raw = doc.get("open_hours")
            thumbnail = doc.get("thumbnail")

            # Parse reviews_per_rating
            rpr = doc.get("reviews_per_rating") or {}
            if not isinstance(rpr, dict):
                rpr = {}

            try:
                lat_f = float(lat) if lat else None
                lng_f = float(lng) if lng else None
            except (ValueError, TypeError):
                lat_f = lng_f = None

            if not title or lat_f is None or lng_f is None:
                continue

            key = (title.lower(), round(lat_f, 4), round(lng_f, 4))
            index[key] = {
                "category_id": cat_id,
                "description": (desc or "").strip() or None,
                "address": (addr or "").strip() or None,
                "google_place_id": google_place_id,
                "google_avg_rating": _safe_float(rating),
                "google_review_count": _safe_int(review_count),
                "google_star_5": _safe_int(rpr.get("5") or rpr.get(5)),
                "google_star_4": _safe_int(rpr.get("4") or rpr.get(4)),
                "google_star_3": _safe_int(rpr.get("3") or rpr.get(3)),
                "google_star_2": _safe_int(rpr.get("2") or rpr.get(2)),
                "google_star_1": _safe_int(rpr.get("1") or rpr.get(1)),
                "phone": (phone or "").strip() or None,
                "website": (website or "").strip() or None,
                "open_hours": str(open_hours_raw)[:500] if open_hours_raw else None,
                "thumbnail": thumbnail,
                "collection": col_name,
            }
            total += 1

    client.close()
    log.info(f"  Indexed {total} google-scrape docs into {len(index)} unique entries")
    return index


def _safe_float(val):
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Migrate public.places → tat.places")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27018",
                        help="MongoDB URI for google-scrape (default: local)")
    args = parser.parse_args()

    # Build google-scrape index
    gs_index = build_google_scrape_index(args.mongo_uri)

    # Connect to PostgreSQL
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    # Get existing tat.places place_ids
    cur.execute("SELECT place_id FROM tat.places")
    existing_ids = set(r[0] for r in cur.fetchall())
    log.info(f"Existing tat.places: {len(existing_ids)}")

    # Get all public.places not in tat.places
    cur.execute("""
        SELECT id, name, name_en, province_id, latitude, longitude,
               rating, thumbnail_url, detail, detail_en
        FROM public.places
        WHERE id NOT IN (SELECT place_id FROM tat.places)
        ORDER BY id
    """)
    new_places = cur.fetchall()
    log.info(f"New places from public.places: {len(new_places)}")

    # Prepare INSERT rows
    rows_to_insert = []
    matched = 0
    unmatched = 0

    for row in new_places:
        pid, name, name_en, prov_id, lat, lng, rating, thumb, detail, detail_en = row

        # Try to find in google-scrape index
        if lat and lng:
            key = ((name or "").lower(), round(float(lat), 4), round(float(lng), 4))
        else:
            key = None

        gs_info = gs_index.get(key) if key else None

        if gs_info:
            matched += 1
            category_id = gs_info["category_id"]
            introduction = gs_info["description"] or detail or detail_en or ""
            address = gs_info["address"]
            google_place_id = gs_info["google_place_id"]
            google_avg_rating = gs_info["google_avg_rating"] or _safe_float(rating)
            google_review_count = gs_info["google_review_count"]
            google_star_5 = gs_info["google_star_5"]
            google_star_4 = gs_info["google_star_4"]
            google_star_3 = gs_info["google_star_3"]
            google_star_2 = gs_info["google_star_2"]
            google_star_1 = gs_info["google_star_1"]
            phone = gs_info["phone"]
            website = gs_info["website"]
            open_hours = gs_info["open_hours"]
            thumbnail_url = gs_info.get("thumbnail") or thumb
        else:
            unmatched += 1
            # Default to "สถานที่อื่นๆ" (13)
            category_id = 13
            introduction = detail or detail_en or ""
            address = None
            google_place_id = None
            google_avg_rating = _safe_float(rating)
            google_review_count = None
            google_star_5 = google_star_4 = google_star_3 = google_star_2 = google_star_1 = None
            phone = None
            website = None
            open_hours = None
            thumbnail_url = thumb

        rows_to_insert.append((
            pid,                 # place_id
            name or "",          # name
            introduction or "",  # introduction
            category_id,         # category_id
            prov_id,             # province_id
            lat,                 # latitude
            lng,                 # longitude
            address,             # address
            google_place_id,     # google_place_id
            google_avg_rating,   # google_avg_rating
            google_review_count, # google_review_count
            google_star_5,       # google_star_5
            google_star_4,       # google_star_4
            google_star_3,       # google_star_3
            google_star_2,       # google_star_2
            google_star_1,       # google_star_1
            phone,               # phone
            website,             # website
            open_hours,          # open_hours
            [thumbnail_url] if thumbnail_url else None,  # thumbnail_urls (array)
        ))

    log.info(f"  Matched with google-scrape: {matched}")
    log.info(f"  Unmatched (default cat=13): {unmatched}")

    if args.dry_run:
        log.info(f"DRY RUN — would insert {len(rows_to_insert)} rows into tat.places")
        # Show sample
        for r in rows_to_insert[:3]:
            log.info(f"  Sample: place_id={r[0]}, name={r[1][:40]}, cat={r[3]}, prov={r[4]}")
    else:
        log.info(f"Inserting {len(rows_to_insert)} rows into tat.places...")
        insert_sql = """
            INSERT INTO tat.places (
                place_id, name, introduction, category_id, province_id,
                latitude, longitude, address, google_place_id,
                google_avg_rating, google_review_count,
                google_star_5, google_star_4, google_star_3, google_star_2, google_star_1,
                phone, website, open_hours, thumbnail_urls
            ) VALUES %s
            ON CONFLICT (place_id) DO NOTHING
        """
        execute_values(cur, insert_sql, rows_to_insert, page_size=1000)
        conn.commit()
        log.info("  INSERT complete!")

    # Verify
    cur.execute("SELECT count(*) FROM tat.places")
    total = cur.fetchone()[0]
    log.info(f"tat.places total after migration: {total}")

    conn.close()
    log.info("DONE")


if __name__ == "__main__":
    main()
