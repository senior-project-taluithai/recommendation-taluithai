#!/usr/bin/env python3
"""
05 — Run Unlabeled Data Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Orchestrates the unlabeled TikTok → TAT entity resolution pipeline:

  Step B: Extract content from all unlabeled videos (NLP pipeline)
  Step C: Score engagement + compute author authority
  Step D: Spatial blocking + fuzzy matching → entity resolution
  Step E: Validate & save results
  Step F: (Optional) Insert resolved mappings into MongoDB

Scraping is handled separately by the TikTok-Content-Scraper repo.

Usage:
    # Full pipeline (use existing MongoDB data)
    python scripts/05_unlabeled_pipeline.py

    # Lower threshold for more matches (include medium confidence)
    python scripts/05_unlabeled_pipeline.py --include-medium

    # Run on a sample first (for debugging)
    python scripts/05_unlabeled_pipeline.py --sample 100

    # Save to MongoDB after resolution
    python scripts/05_unlabeled_pipeline.py --save-mongo
"""

import argparse
import json
import logging
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "unlabeled_pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Unlabeled TikTok → TAT Entity Resolution Pipeline",
    )

    # Resolution options
    parser.add_argument("--include-medium", action="store_true",
                        help="Include medium-confidence matches (75-90%%)")
    parser.add_argument("--include-low", action="store_true",
                        help="Include low-confidence matches (60-75%%) for review")
    parser.add_argument("--threshold", type=float, default=90.0,
                        help="Auto-accept threshold (default: 90)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only N videos (for debugging)")
    parser.add_argument("--include-labeled", action="store_true",
                        help="Also process already-labeled videos (default: skip)")

    # Output options
    parser.add_argument("--save-mongo", action="store_true",
                        help="Insert resolved matches into MongoDB")
    parser.add_argument("--mongo-collection", type=str, default="place_videos_unlabeled",
                        help="MongoDB collection name for resolved matches")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/)")

    args = parser.parse_args()

    t_start = time.time()
    output_dir = DATA_DIR if args.output_dir is None else __import__("pathlib").Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Steps B-E: Entity Resolution
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("Running Entity Resolution Pipeline")
    logger.info("=" * 70)

    from src.unlabeled.resolver import resolve_videos, save_report, insert_to_mongodb

    # Optionally limit to a sample
    video_ids = None
    if args.sample:
        logger.info(f"  Sampling {args.sample} videos for debugging...")
        from pymongo import MongoClient
        from config import MONGO_URI, MONGO_TIKTOK_DB

        client = MongoClient(MONGO_URI)
        db = client[MONGO_TIKTOK_DB]
        sample_docs = list(
            db.content_metadata.aggregate([{"$sample": {"size": args.sample}}])
        )
        video_ids = [str(d["video_id"]) for d in sample_docs]
        client.close()
        logger.info(f"  Sampled {len(video_ids)} video_ids")

    report = resolve_videos(
        video_ids=video_ids,
        exclude_labeled=not args.include_labeled,
        match_threshold=args.threshold,
        include_medium=args.include_medium,
        include_low=args.include_low,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step E: Save results
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("Saving results...")
    save_report(report, output_dir=output_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Step F: Insert to MongoDB (optional)
    # ══════════════════════════════════════════════════════════════════════════
    if args.save_mongo:
        logger.info("=" * 70)
        logger.info(f"Inserting to MongoDB collection: {args.mongo_collection}")
        insert_to_mongodb(report, collection_name=args.mongo_collection)

    # ══════════════════════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Resolved {len(report.resolved_matches)} video → place matches")
    logger.info(f"  High: {report.total_high_confidence} | "
                f"Medium: {report.total_medium_confidence} | "
                f"Low: {report.total_low_confidence}")

    # Print a few sample matches
    if report.resolved_matches:
        logger.info("\nSample resolved matches:")
        for m in report.resolved_matches[:10]:
            logger.info(
                f"  [{m.confidence.upper():6s}] {m.match_score:5.1f}% "
                f"video={m.video_id[:12]}... → {m.place_name} "
                f"(token='{m.matched_token}', spatial={m.is_spatially_blocked})"
            )


if __name__ == "__main__":
    main()
