"""
Entity Resolution — Full pipeline: video → place assignment via spatial blocking + fuzzy match
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Combines all modules to resolve unlabeled TikTok videos to TAT places:
  1. Content extraction (NLP)
  2. Engagement scoring
  3. Spatial blocking
  4. Fuzzy text matching
  5. Validation (threshold-based)

Produces a mapping: video_id → place_id with confidence scores.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_DIR

from .extractor import ExtractedContent, extract_batch_from_mongo
from .scorer import ScoredVideo, score_batch, compute_author_stats_from_batch
from .spatial_blocker import SpatialBlocker, LocationBlock
from .fuzzy_matcher import match_video_to_places, MatchResult

logger = logging.getLogger(__name__)


# ─── Thresholds ───────────────────────────────────────────────────────────────
MATCH_THRESHOLD_HIGH = 90.0      # ≥90% → auto-accept (confident match)
MATCH_THRESHOLD_MEDIUM = 75.0    # 75-90% → accept if spatial blocking agrees
MATCH_THRESHOLD_LOW = 60.0       # 60-75% → manual review recommended
MIN_ENGAGEMENT_SCORE = 0.05      # filter videos with near-zero engagement


@dataclass
class ResolvedMatch:
    """A validated video → place assignment."""
    video_id: str
    author_id: str
    place_id: int
    place_name: str
    match_score: float           # 0-100
    matched_token: str           # the keyword that matched
    match_strategy: str          # fuzzy match strategy used
    confidence: str              # "high" | "medium" | "low"
    engagement_score: float      # 0-1
    preference_score: float      # 0-1
    is_spatially_blocked: bool   # was spatial blocking used?
    province_id: int | None = None
    matched_terms: list[str] = field(default_factory=list)  # spatial block terms


@dataclass
class ResolutionReport:
    """Summary of an entity resolution run."""
    total_videos_processed: int = 0
    total_videos_scored: int = 0
    total_high_confidence: int = 0
    total_medium_confidence: int = 0
    total_low_confidence: int = 0
    total_no_match: int = 0
    total_spatially_blocked: int = 0
    resolved_matches: list[ResolvedMatch] = field(default_factory=list)


def resolve_videos(
    video_ids: list[str] | None = None,
    exclude_labeled: bool = True,
    match_threshold: float = MATCH_THRESHOLD_HIGH,
    include_medium: bool = True,
    include_low: bool = False,
    max_candidates_per_video: int = 2000,
    batch_size: int = 5000,
) -> ResolutionReport:
    """
    Full entity resolution pipeline.

    Args:
        video_ids:            specific videos to process (None = all unlabeled)
        exclude_labeled:      skip videos already in place_videos
        match_threshold:      minimum score for auto-accept (default 90)
        include_medium:       include 75-90% matches (requires spatial agreement)
        include_low:          include 60-75% matches (for review)
        max_candidates_per_video: max places to fuzzy-match against per video
        batch_size:           MongoDB fetch batch size

    Returns:
        ResolutionReport with all resolved matches and statistics.
    """
    report = ResolutionReport()

    # ── Step 1: Extract content ──
    logger.info("=" * 70)
    logger.info("Step 1: Extracting content from MongoDB...")
    contents = extract_batch_from_mongo(
        video_ids=video_ids,
        exclude_labeled=exclude_labeled,
        batch_size=batch_size,
    )
    report.total_videos_processed = len(contents)
    logger.info(f"  Extracted {len(contents)} videos")

    if not contents:
        logger.info("  No videos to process.")
        return report

    # ── Step 2: Score engagement ──
    logger.info("=" * 70)
    logger.info("Step 2: Scoring engagement...")
    author_stats = compute_author_stats_from_batch(contents)
    scored = score_batch(contents, author_stats, min_engagement=MIN_ENGAGEMENT_SCORE)
    report.total_videos_scored = len(scored)
    logger.info(f"  {len(scored)} videos passed engagement threshold")

    # ── Step 3: Spatial blocking ──
    logger.info("=" * 70)
    logger.info("Step 3: Loading spatial blocker...")
    blocker = SpatialBlocker()
    blocker.load_gazetteer()

    # ── Step 4 & 5: For each scored video → spatial block → fuzzy match → validate ──
    logger.info("=" * 70)
    logger.info("Step 4-5: Entity resolution (spatial blocking + fuzzy matching)...")

    from tqdm import tqdm

    for sv in tqdm(scored, desc="resolving"):
        content = sv.content
        if content is None:
            continue

        # Spatial blocking
        block = blocker.match_location(
            hashtags=content.hashtags,
            suggest_words=content.suggest_words,
            location_tokens=content.location_tokens,
        )

        if block.is_blocked:
            report.total_spatially_blocked += 1

        # Get candidate places
        candidates = blocker.get_candidate_places(block, max_candidates=max_candidates_per_video)

        if not candidates:
            report.total_no_match += 1
            continue

        # Fuzzy matching
        matches = match_video_to_places(
            suggest_words=content.suggest_words,
            hashtags=content.hashtags,
            sticker_texts=content.sticker_texts,
            candidate_places=candidates,
            top_k=3,
            min_score=MATCH_THRESHOLD_LOW if include_low else MATCH_THRESHOLD_MEDIUM,
        )

        if not matches:
            report.total_no_match += 1
            continue

        best = matches[0]

        # ── Validation ──
        confidence: str
        accept: bool

        if best.match_score >= MATCH_THRESHOLD_HIGH:
            confidence = "high"
            accept = True
            report.total_high_confidence += 1

        elif best.match_score >= MATCH_THRESHOLD_MEDIUM:
            confidence = "medium"
            # Accept medium only if spatial blocking supports it
            accept = include_medium and (
                block.is_blocked or best.match_score >= 85.0
            )
            if accept:
                report.total_medium_confidence += 1
            else:
                report.total_no_match += 1
                continue

        elif best.match_score >= MATCH_THRESHOLD_LOW:
            confidence = "low"
            accept = include_low
            if accept:
                report.total_low_confidence += 1
            else:
                report.total_no_match += 1
                continue
        else:
            report.total_no_match += 1
            continue

        if accept:
            resolved = ResolvedMatch(
                video_id=content.video_id,
                author_id=content.author_id,
                place_id=best.place_id,
                place_name=best.place_name,
                match_score=best.match_score,
                matched_token=best.matched_token,
                match_strategy=best.strategy,
                confidence=confidence,
                engagement_score=sv.engagement_score,
                preference_score=sv.preference_score,
                is_spatially_blocked=block.is_blocked,
                province_id=best.province_id,
                matched_terms=block.matched_terms,
            )
            report.resolved_matches.append(resolved)

    # ── Summary ──
    logger.info("=" * 70)
    logger.info("Entity Resolution Summary:")
    logger.info(f"  Total processed:       {report.total_videos_processed}")
    logger.info(f"  Passed engagement:     {report.total_videos_scored}")
    logger.info(f"  Spatially blocked:     {report.total_spatially_blocked}")
    logger.info(f"  High confidence (≥90): {report.total_high_confidence}")
    logger.info(f"  Medium (75-90):        {report.total_medium_confidence}")
    logger.info(f"  Low (60-75):           {report.total_low_confidence}")
    logger.info(f"  No match:              {report.total_no_match}")
    logger.info(f"  Total resolved:        {len(report.resolved_matches)}")

    return report


def save_report(report: ResolutionReport, output_dir: Path | None = None) -> Path:
    """
    Save resolution report to JSON.
    Also saves a place_videos-compatible mapping for MongoDB insertion.
    """
    out_dir = output_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full report
    report_path = out_dir / "entity_resolution_report.json"
    report_data = {
        "summary": {
            "total_videos_processed": report.total_videos_processed,
            "total_videos_scored": report.total_videos_scored,
            "total_high_confidence": report.total_high_confidence,
            "total_medium_confidence": report.total_medium_confidence,
            "total_low_confidence": report.total_low_confidence,
            "total_no_match": report.total_no_match,
            "total_spatially_blocked": report.total_spatially_blocked,
            "total_resolved": len(report.resolved_matches),
        },
        "matches": [asdict(m) for m in report.resolved_matches],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # place_videos-compatible format: place_id → [video_ids]
    place_videos_map: dict[int, list[str]] = {}
    for m in report.resolved_matches:
        place_videos_map.setdefault(m.place_id, []).append(m.video_id)

    pv_path = out_dir / "place_videos_unlabeled.json"
    with open(pv_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"place_id": pid, "video_ids": vids} for pid, vids in place_videos_map.items()],
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"  Report saved: {report_path}")
    logger.info(f"  Place-videos mapping: {pv_path}")
    logger.info(f"  Unique places matched: {len(place_videos_map)}")

    return report_path


def insert_to_mongodb(report: ResolutionReport, collection_name: str = "place_videos_unlabeled"):
    """
    Insert resolved matches into MongoDB as a new collection.
    Format matches `place_videos` collection: {place_id, video_ids[]}.
    """
    from pymongo import MongoClient
    from config import MONGO_URI, MONGO_TIKTOK_DB

    # Build place → videos mapping
    place_videos_map: dict[int, list[str]] = {}
    for m in report.resolved_matches:
        place_videos_map.setdefault(m.place_id, []).append(m.video_id)

    if not place_videos_map:
        logger.info("  No matches to insert.")
        return

    client = MongoClient(MONGO_URI)
    db = client[MONGO_TIKTOK_DB]
    coll = db[collection_name]

    # Use upsert: $addToSet to append new video_ids
    from pymongo import UpdateOne
    ops = []
    for pid, vids in place_videos_map.items():
        ops.append(
            UpdateOne(
                {"place_id": pid},
                {"$addToSet": {"video_ids": {"$each": vids}}},
                upsert=True,
            )
        )

    result = coll.bulk_write(ops, ordered=False)
    client.close()

    logger.info(
        f"  MongoDB: {result.upserted_count} new, "
        f"{result.modified_count} updated in '{collection_name}'"
    )
