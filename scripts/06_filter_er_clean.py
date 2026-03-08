#!/usr/bin/env python3
"""
Re-filter entity resolution matches and rebuild place_videos_unlabeled
collection in MongoDB with only clean data.

Criteria: high confidence + spatially blocked + token length > 3
"""
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, MONGO_URI, MONGO_TIKTOK_DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIN_TOKEN_LEN = 4  # exclude tokens <= 3 chars

def main():
    # ── Load report ──
    report_path = DATA_DIR / "entity_resolution_report.json"
    logger.info(f"Loading report: {report_path}")
    with open(report_path) as f:
        report = json.load(f)

    matches = report["matches"]
    logger.info(f"Total matches in report: {len(matches)}")

    # ── Filter: high confidence + spatially blocked + token > 3 chars ──
    clean = [
        m for m in matches
        if m["confidence"] == "high"
        and m.get("is_spatially_blocked")
        and len(m["matched_token"]) > 3
    ]
    logger.info(f"After filter (high + spatial + token>{MIN_TOKEN_LEN - 1}): {len(clean)} matches")

    # Build place → videos mapping
    place_videos: dict[int, list[str]] = {}
    for m in clean:
        place_videos.setdefault(m["place_id"], []).append(m["video_id"])

    # Deduplicate video_ids per place
    place_videos = {pid: list(set(vids)) for pid, vids in place_videos.items()}
    total_pairs = sum(len(v) for v in place_videos.values())
    logger.info(f"Unique places: {len(place_videos)}, total video-place pairs: {total_pairs}")

    # ── Drop and rebuild MongoDB collection ──
    from pymongo import MongoClient, UpdateOne

    client = MongoClient(MONGO_URI)
    db = client[MONGO_TIKTOK_DB]
    coll = db["place_videos_unlabeled"]

    # Drop old collection entirely
    coll.drop()
    logger.info("Dropped old place_videos_unlabeled collection")

    # Insert clean data
    docs = [{"place_id": pid, "video_ids": vids} for pid, vids in place_videos.items()]
    if docs:
        coll.insert_many(docs)
    logger.info(f"Inserted {len(docs)} clean place-video documents")

    # ── Also save as local JSON ──
    # Strip MongoDB _id fields for JSON serialization
    clean_docs = [{"place_id": d["place_id"], "video_ids": d["video_ids"]} for d in docs]
    pv_path = DATA_DIR / "place_videos_unlabeled.json"
    with open(pv_path, "w", encoding="utf-8") as f:
        json.dump(clean_docs, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved filtered mapping: {pv_path}")

    client.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
