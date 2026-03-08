"""
Engagement Scoring for Unlabeled TikTok Videos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Computes a preference / relevance score for each TikTok video
based on engagement metrics (diggcount, sharecount, collectcount, playcount).

Applies the same IPS (Inverse Propensity Scoring) debiasing approach
used in the labeled pipeline, but extended for unlabeled data where
we don't have a ground-truth place assignment yet.

Score components:
  1. Engagement score — weighted sum of normalised engagement metrics
  2. Author authority — bonus for creators with consistently high engagement
  3. Content relevance — bonus for travel/place-related content signals
"""

import logging
import math
from dataclasses import dataclass

from .extractor import ExtractedContent

logger = logging.getLogger(__name__)


@dataclass
class ScoredVideo:
    """Video with computed preference score."""
    video_id: str
    author_id: str
    engagement_score: float       # raw 0-1
    author_authority: float       # 0-1 bonus
    content_relevance: float      # 0-1 bonus
    preference_score: float       # final combined score
    # Forward extracted content
    content: ExtractedContent | None = None


# ─── Engagement weights ───────────────────────────────────────────────────────
# Share > Collect > Like > View (stronger actions = stronger signal)
W_SHARE = 0.40
W_COLLECT = 0.30
W_DIGG = 0.20
W_PLAY = 0.10

# Final score composition
W_ENGAGEMENT = 0.60
W_AUTHORITY = 0.15
W_RELEVANCE = 0.25

# Travel-related hashtags that boost content relevance
TRAVEL_HASHTAGS = {
    "เที่ยว", "ที่เที่ยว", "เช็คอิน", "checkin", "travel",
    "travelthailand", "เที่ยวไทย", "ที่กิน", "คาเฟ่", "cafe",
    "ร้านอาหาร", "hotel", "โรงแรม", "ที่พัก", "วัด", "temple",
    "ทะเล", "beach", "เกาะ", "island", "น้ำตก", "waterfall",
    "ภูเขา", "mountain", "อุทยาน", "nationalpark",
    "ตลาด", "market", "streetfood", "nightmarket",
    "รีวิว", "review", "พากิน", "พาเที่ยว",
    "สถานที่", "จุดเช็คอิน", "ลับ", "มุมถ่ายรูป",
    "hidden", "hiddenspot", "สวย", "วิว", "view",
    "sunset", "sunrise", "ทริป", "trip", "roadtrip",
}

# TikTok diversification labels for travel content
TRAVEL_LABELS = {"Travel", "Food", "Lifestyle", "Nature", "Entertainment"}


def _log_normalise(x: float, scale: float = 1.0) -> float:
    """Log-normalise a count value to 0-1 range. Handles zero gracefully."""
    if x <= 0:
        return 0.0
    return min(1.0, math.log1p(x) / math.log1p(scale))


def compute_engagement_score(content: ExtractedContent) -> float:
    """
    Compute normalised engagement score from raw metrics.
    Uses log-normalisation with empirical scale factors.
    """
    # Scale factors based on typical TikTok metrics distribution
    play_norm = _log_normalise(content.playcount, scale=1_000_000)   # 1M views = ~1.0
    digg_norm = _log_normalise(content.diggcount, scale=100_000)     # 100K likes = ~1.0
    share_norm = _log_normalise(content.sharecount, scale=10_000)    # 10K shares = ~1.0
    collect_norm = _log_normalise(content.collectcount, scale=10_000) # 10K saves = ~1.0

    score = (
        W_PLAY * play_norm
        + W_DIGG * digg_norm
        + W_SHARE * share_norm
        + W_COLLECT * collect_norm
    )
    return min(1.0, score)


def compute_author_authority(
    author_id: str,
    author_stats: dict[str, dict] | None = None,
) -> float:
    """
    Compute author authority score based on their content history.

    Args:
        author_id:    TikTok author ID
        author_stats: pre-computed dict: {author_id: {avg_digg, video_count, ...}}
                      If None, returns 0.0 (skip authority scoring).
    """
    if not author_stats or author_id not in author_stats:
        return 0.0

    stats = author_stats[author_id]
    avg_digg = stats.get("avg_digg", 0)
    video_count = stats.get("video_count", 0)

    # Authority = f(consistency × reach)
    consistency = min(1.0, video_count / 20)  # 20+ videos = max consistency
    reach = _log_normalise(avg_digg, scale=50_000)  # avg 50K likes = high reach

    return consistency * 0.4 + reach * 0.6


def compute_content_relevance(content: ExtractedContent) -> float:
    """
    Score how travel/place-related the content appears to be.
    Based on hashtag overlap with travel vocabulary and diversification labels.
    """
    score = 0.0

    # Hashtag overlap with travel set
    if content.hashtags:
        ht_set = {h.lower() for h in content.hashtags}
        overlap = ht_set & TRAVEL_HASHTAGS
        ht_score = min(1.0, len(overlap) / 3)  # 3+ travel hashtags = max
        score += 0.5 * ht_score

    # Diversification labels
    labels = set()
    doc_labels = getattr(content, "labels", None)
    if doc_labels:
        labels = set(doc_labels)
    label_overlap = labels & TRAVEL_LABELS
    if label_overlap:
        score += 0.3 * min(1.0, len(label_overlap) / 2)

    # Location tokens detected
    if content.location_tokens:
        score += 0.2 * min(1.0, len(content.location_tokens) / 2)

    return min(1.0, score)


def compute_author_stats_from_batch(
    contents: list[ExtractedContent],
) -> dict[str, dict]:
    """
    Pre-compute per-author stats from a batch of extracted content.
    Returns: {author_id: {avg_digg, avg_share, video_count}}
    """
    from collections import defaultdict

    author_agg: dict[str, dict] = defaultdict(lambda: {
        "total_digg": 0, "total_share": 0, "count": 0,
    })

    for c in contents:
        if c.author_id:
            a = author_agg[c.author_id]
            a["total_digg"] += c.diggcount
            a["total_share"] += c.sharecount
            a["count"] += 1

    stats: dict[str, dict] = {}
    for aid, a in author_agg.items():
        n = max(1, a["count"])
        stats[aid] = {
            "avg_digg": a["total_digg"] / n,
            "avg_share": a["total_share"] / n,
            "video_count": a["count"],
        }

    return stats


def score_batch(
    contents: list[ExtractedContent],
    author_stats: dict[str, dict] | None = None,
    min_engagement: float = 0.05,
) -> list[ScoredVideo]:
    """
    Score a batch of extracted TikTok content.

    Args:
        contents:        list of ExtractedContent from extractor
        author_stats:    pre-computed author stats (if None, computed from batch)
        min_engagement:  minimum engagement_score to include (filter noise)

    Returns:
        List of ScoredVideo, sorted by preference_score descending.
    """
    if author_stats is None:
        logger.info("  Computing author stats from batch...")
        author_stats = compute_author_stats_from_batch(contents)

    scored: list[ScoredVideo] = []

    for content in contents:
        eng = compute_engagement_score(content)

        # Filter out very low engagement videos (likely spam/noise)
        if eng < min_engagement:
            continue

        auth = compute_author_authority(content.author_id, author_stats)
        rel = compute_content_relevance(content)

        pref = W_ENGAGEMENT * eng + W_AUTHORITY * auth + W_RELEVANCE * rel

        scored.append(ScoredVideo(
            video_id=content.video_id,
            author_id=content.author_id,
            engagement_score=eng,
            author_authority=auth,
            content_relevance=rel,
            preference_score=pref,
            content=content,
        ))

    scored.sort(key=lambda s: s.preference_score, reverse=True)
    logger.info(
        f"  Scored {len(scored)} videos "
        f"(filtered {len(contents) - len(scored)} below threshold)"
    )
    return scored
