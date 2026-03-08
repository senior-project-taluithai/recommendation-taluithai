"""
Fuzzy Text Matching — Compare TikTok suggest words with TAT place names
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Multi-strategy fuzzy matching between TikTok-extracted keywords and
TAT place names, using rapidfuzz for high-performance string similarity.

Strategies (combined score):
  1. Ratio          — overall Levenshtein-based similarity
  2. Partial ratio  — best substring match (handles partial place names)
  3. Token sort     — order-independent token matching
  4. Token set      — handles extra/missing tokens gracefully
  5. Thai normalisation — strip common prefixes (วัด, ร้าน, บ้าน, etc.)

Requirements:
    pip install rapidfuzz pythainlp
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import Levenshtein
    HAS_RAPIDFUZZ = True
except ImportError:
    logger.warning("rapidfuzz not installed. Install with: pip install rapidfuzz")
    HAS_RAPIDFUZZ = False

try:
    from pythainlp.util import normalize as thai_normalize
    HAS_PYTHAINLP = True
except ImportError:
    HAS_PYTHAINLP = False

    def thai_normalize(text: str) -> str:
        return text


# ─── Thai place name prefixes to strip for matching ───────────────────────────
THAI_PLACE_PREFIXES = [
    "วัด", "ร้าน", "บ้าน", "ตลาด", "ถนน", "สวน", "หาด",
    "อุทยานแห่งชาติ", "เขื่อน", "น้ำตก", "ถ้ำ", "ดอย",
    "ลาน", "ศูนย์", "พิพิธภัณฑ์", "สนามบิน", "สถานี",
    "โรงแรม", "รีสอร์ท", "เกาะ", "แหลม", "อ่าว", "คลอง",
]

# Compile prefix regex (match at start of string)
_PREFIX_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(p) for p in sorted(THAI_PLACE_PREFIXES, key=len, reverse=True)) + r")\s*",
)


@dataclass
class MatchResult:
    """Result of fuzzy matching a video against TAT places."""
    place_id: int
    place_name: str
    match_score: float            # 0-100 percentage
    matched_token: str            # which suggest word triggered the match
    strategy: str                 # which matching strategy produced the score
    province_id: int | None = None
    district_id: int | None = None


@dataclass
class VideoMatchResults:
    """All match results for a single video."""
    video_id: str
    matches: list[MatchResult] = field(default_factory=list)
    best_match: MatchResult | None = None


def _normalise_for_matching(text: str) -> str:
    """
    Normalise a text string for fuzzy matching:
    - Thai character normalisation
    - Lowercase
    - Strip common place prefixes
    - Remove extra whitespace
    """
    if not text:
        return ""

    text = text.strip()
    if HAS_PYTHAINLP:
        text = thai_normalize(text)

    text = text.lower()

    # Remove parenthetical content (e.g. "น้ำตกเพียงดิน (น้ำตกวิสุทธารา)")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _strip_prefix(text: str) -> str:
    """Strip common Thai place prefixes for core-name matching."""
    return _PREFIX_PATTERN.sub("", text).strip()


def _compute_match_score(query: str, target: str) -> tuple[float, str]:
    """
    Compute multi-strategy fuzzy match score between query and target.

    Returns:
        (score 0-100, strategy_name)
    """
    if not HAS_RAPIDFUZZ:
        # Fallback: exact containment check
        if query in target or target in query:
            return (95.0, "containment")
        return (0.0, "none")

    q = _normalise_for_matching(query)
    t = _normalise_for_matching(target)

    if not q or not t:
        return (0.0, "none")

    # Strategy 1: Full ratio
    score_ratio = fuzz.ratio(q, t)

    # Strategy 2: Partial ratio (best substring match)
    score_partial = fuzz.partial_ratio(q, t)

    # Strategy 3: Token sort ratio
    score_token_sort = fuzz.token_sort_ratio(q, t)

    # Strategy 4: Token set ratio (handles extra tokens)
    score_token_set = fuzz.token_set_ratio(q, t)

    # Strategy 5: Prefix-stripped matching
    q_stripped = _strip_prefix(q)
    t_stripped = _strip_prefix(t)
    score_stripped = 0.0
    if q_stripped and t_stripped:
        score_stripped = fuzz.ratio(q_stripped, t_stripped)

    # Weighted combination — partial + token_set are best for Thai place names
    scores = {
        "ratio": score_ratio,
        "partial": score_partial,
        "token_sort": score_token_sort,
        "token_set": score_token_set,
        "stripped": score_stripped,
    }

    # Use the maximum score (best strategy wins)
    best_strategy = max(scores, key=scores.get)
    best_score = scores[best_strategy]

    return (best_score, best_strategy)


def match_video_to_places(
    suggest_words: list[str],
    hashtags: list[str],
    sticker_texts: list[str],
    candidate_places: list[dict],
    top_k: int = 5,
    min_score: float = 50.0,
) -> list[MatchResult]:
    """
    Match a video's extracted keywords against candidate TAT places.

    Args:
        suggest_words:     ranked keywords from content extractor
        hashtags:          video hashtags
        sticker_texts:     OCR sticker texts
        candidate_places:  list of dicts: [{place_id, name, province_id, district_id}]
        top_k:             max results to return
        min_score:         minimum match score (0-100) to include

    Returns:
        List of MatchResult sorted by match_score descending.
    """
    if not candidate_places:
        return []

    # Build a flat list of query tokens to try
    query_tokens: list[str] = []

    # Priority 1: sticker texts (most reliable for place names)
    for st in sticker_texts:
        st_clean = st.strip()
        if len(st_clean) >= 2:
            query_tokens.append(st_clean)

    # Priority 2: hashtags (often contain place names as compounds)
    for h in hashtags:
        h_clean = h.strip()
        if len(h_clean) >= 3:
            query_tokens.append(h_clean)

    # Priority 3: suggest words (NLP extracted, may be noisy)
    for sw in suggest_words[:15]:  # top-15 suggest words only
        if len(sw) >= 3:
            query_tokens.append(sw)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for t in query_tokens:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique_tokens.append(t)

    if not unique_tokens:
        return []

    # Build place name lookup
    place_lookup: dict[str, dict] = {}
    for p in candidate_places:
        name = p.get("name", "").strip()
        if name:
            place_lookup[name] = p

    place_names = list(place_lookup.keys())

    if not place_names:
        return []

    # ── Multi-token matching ──
    best_per_place: dict[int, MatchResult] = {}

    if HAS_RAPIDFUZZ:
        for token in unique_tokens:
            # Use rapidfuzz.process.extract for batch matching
            matches = process.extract(
                token,
                place_names,
                scorer=fuzz.token_set_ratio,
                limit=min(top_k * 2, 20),
                score_cutoff=min_score,
            )

            for match_name, score, _ in matches:
                # Refine with multi-strategy scoring
                refined_score, strategy = _compute_match_score(token, match_name)
                final_score = max(score, refined_score)

                place_info = place_lookup[match_name]
                place_id = place_info["place_id"]

                if place_id not in best_per_place or final_score > best_per_place[place_id].match_score:
                    best_per_place[place_id] = MatchResult(
                        place_id=place_id,
                        place_name=match_name,
                        match_score=final_score,
                        matched_token=token,
                        strategy=strategy,
                        province_id=place_info.get("province_id"),
                        district_id=place_info.get("district_id"),
                    )
    else:
        # Fallback without rapidfuzz: simple containment matching
        for token in unique_tokens:
            t_lower = token.lower()
            for name, place_info in place_lookup.items():
                n_lower = name.lower()
                # Simple containment check
                if t_lower in n_lower or n_lower in t_lower:
                    score = 95.0 if n_lower == t_lower else 85.0
                    place_id = place_info["place_id"]
                    if place_id not in best_per_place or score > best_per_place[place_id].match_score:
                        best_per_place[place_id] = MatchResult(
                            place_id=place_id,
                            place_name=name,
                            match_score=score,
                            matched_token=token,
                            strategy="containment",
                            province_id=place_info.get("province_id"),
                            district_id=place_info.get("district_id"),
                        )

    # Sort by score, take top-K
    results = sorted(best_per_place.values(), key=lambda r: r.match_score, reverse=True)
    return results[:top_k]


def batch_match(
    videos: list[dict[str, Any]],
    candidate_places: list[dict],
    top_k: int = 5,
    min_score: float = 50.0,
) -> list[VideoMatchResults]:
    """
    Match a batch of videos against candidate places.

    Args:
        videos: list of dicts with keys:
                {video_id, suggest_words, hashtags, sticker_texts}
        candidate_places: TAT places [{place_id, name, ...}]
        top_k:     max matches per video
        min_score: minimum score threshold

    Returns:
        List of VideoMatchResults.
    """
    results: list[VideoMatchResults] = []

    for v in videos:
        matches = match_video_to_places(
            suggest_words=v.get("suggest_words", []),
            hashtags=v.get("hashtags", []),
            sticker_texts=v.get("sticker_texts", []),
            candidate_places=candidate_places,
            top_k=top_k,
            min_score=min_score,
        )

        vmr = VideoMatchResults(
            video_id=v["video_id"],
            matches=matches,
            best_match=matches[0] if matches else None,
        )
        results.append(vmr)

    return results
