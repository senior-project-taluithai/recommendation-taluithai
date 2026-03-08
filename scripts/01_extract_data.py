#!/usr/bin/env python3
"""
Step 1: Extract data from PostgreSQL + MongoDB → save to data/ folder.

Run this ONCE on a machine with DB access (before uploading to vast.ai).

Usage:
    python scripts/01_extract_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.extract import (
    extract_places,
    extract_categories,
    extract_provinces,
    extract_place_videos,
    extract_tiktok_videos,
    extract_graph_data,
    compute_propensity_scores,
    save_extracted_data,
)
from config import DATA_DIR
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("STEP 1: Data Extraction Pipeline")
    logger.info("=" * 60)
    
    # 1. PostgreSQL
    places_df = extract_places()
    categories_df = extract_categories()
    provinces_df = extract_provinces()
    
    # Save categories and provinces separately
    categories_df.to_parquet(DATA_DIR / "categories.parquet", index=False)
    provinces_df.to_parquet(DATA_DIR / "provinces.parquet", index=False)
    
    # 2. MongoDB: place → videos mapping
    place_to_videos = extract_place_videos()
    
    # 3. MongoDB: TikTok video metadata → training pairs
    training_pairs_df = extract_tiktok_videos(place_to_videos)
    
    # 4. Graph data for GNN
    graph_data = extract_graph_data(place_to_videos)
    
    # 5. Propensity scores
    propensity_scores = compute_propensity_scores(places_df, place_to_videos)
    
    # 6. Save everything
    save_extracted_data(
        places_df, training_pairs_df, propensity_scores,
        graph_data, place_to_videos,
    )
    
    logger.info("=" * 60)
    logger.info("Data extraction complete!")
    logger.info(f"All files saved to: {DATA_DIR}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Upload data/ folder to vast.ai")
    logger.info("  2. Run: python scripts/02_train_gnn.py")
    logger.info("  3. Run: python scripts/03_train_two_tower.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
