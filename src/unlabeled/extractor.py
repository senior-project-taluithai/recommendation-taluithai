"""
Content Extraction — NLP pipeline for TikTok video text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extracts structured information from raw TikTok video metadata:
  1. Thai word segmentation (pythainlp)
  2. Hashtag extraction & normalisation
  3. Sticker text extraction (OCR overlays)
  4. Named-entity-like extraction (location tokens, place-name candidates)
  5. "Suggest words" — ranked keywords that could identify a place

Requirements:
    pip install pythainlp rapidfuzz
"""

import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import MONGO_URI, MONGO_TIKTOK_DB

logger = logging.getLogger(__name__)

# ─── Thai NLP setup ───────────────────────────────────────────────────────────
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import normalize as thai_normalize
    from pythainlp.corpus import thai_stopwords

    THAI_STOPWORDS = thai_stopwords()
    HAS_PYTHAINLP = True
except ImportError:
    logger.warning(
        "pythainlp not installed — Thai NLP segmentation disabled.\n"
        "Install with: pip install pythainlp"
    )
    HAS_PYTHAINLP = False
    THAI_STOPWORDS = set()

    def word_tokenize(text: str, engine: str = "newmm") -> list[str]:
        """Fallback: simple whitespace split."""
        return text.split()

    def thai_normalize(text: str) -> str:
        return text


# ─── Regex patterns ───────────────────────────────────────────────────────────
RE_HASHTAG = re.compile(r"#([\wก-๙]+)")
RE_AT_MENTION = re.compile(r"@[\w.]+")
RE_URL = re.compile(r"https?://\S+")
RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
# Thai province / district marker words
RE_LOCATION_PREFIX = re.compile(
    r"(จังหวัด|จ\.|อำเภอ|อ\.|ตำบล|ต\.|เขต|แขวง|"
    r"จ\.(?=\s*[\u0E01-\u0E4F])|อ\.(?=\s*[\u0E01-\u0E4F]))"
)


@dataclass
class ExtractedContent:
    """Structured extraction result for a single TikTok video."""
    video_id: str
    author_id: str

    # Raw fields
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    sticker_texts: list[str] = field(default_factory=list)

    # NLP-derived
    tokens: list[str] = field(default_factory=list)         # segmented tokens
    clean_tokens: list[str] = field(default_factory=list)    # after stopword removal
    suggest_words: list[str] = field(default_factory=list)   # ranked keyword candidates
    location_tokens: list[str] = field(default_factory=list) # province/district mentions

    # Engagement (raw)
    playcount: int = 0
    diggcount: int = 0
    sharecount: int = 0
    collectcount: int = 0


def _clean_text(text: str) -> str:
    """Remove URLs, mentions, emojis; normalise Thai characters."""
    text = RE_URL.sub("", text)
    text = RE_AT_MENTION.sub("", text)
    text = RE_EMOJI.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if HAS_PYTHAINLP:
        text = thai_normalize(text)
    return text


def _extract_hashtags_from_text(text: str) -> list[str]:
    """Pull #hashtag tokens from description text."""
    return [h.strip() for h in RE_HASHTAG.findall(text) if h.strip()]


def _segment_thai(text: str) -> list[str]:
    """Word-segment Thai text using pythainlp newmm engine."""
    # Strip hashtag symbols before segmentation
    text_no_hash = RE_HASHTAG.sub(r"\1", text)
    text_clean = _clean_text(text_no_hash)
    if not text_clean:
        return []
    tokens = word_tokenize(text_clean, engine="newmm")
    tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
    return tokens


def _extract_location_tokens(tokens: list[str], text: str) -> list[str]:
    """
    Identify tokens that look like location names.
    Heuristic: tokens following จังหวัด/อำเภอ/ตำบล prefixes,
    or Thai proper-noun-like tokens (multi-syllable, not common words).
    """
    locations: list[str] = []

    # Strategy 1: look for "จังหวัดXXX" / "อำเภอXXX" patterns in raw text
    for m in RE_LOCATION_PREFIX.finditer(text):
        start = m.end()
        # Grab the next token(s) after the prefix
        rest = text[start:start + 30].strip()
        if rest:
            loc_tokens = word_tokenize(rest, engine="newmm") if HAS_PYTHAINLP else rest.split()
            if loc_tokens:
                # Take up to 2 tokens as the location name
                loc = "".join(loc_tokens[:2]).strip()
                if len(loc) >= 2:
                    locations.append(loc)

    # Strategy 2: tokens that are in Thai and ≥3 chars (potential proper nouns)
    thai_pattern = re.compile(r"^[\u0E01-\u0E4F]+$")
    for t in tokens:
        if (
            thai_pattern.match(t)
            and len(t) >= 3
            and t not in THAI_STOPWORDS
            and t not in {"เที่ยว", "สถานที่", "ร้าน", "อาหาร", "ที่พัก", "โรงแรม",
                          "คาเฟ่", "ทะเล", "น้ำตก", "วัด", "ภูเขา", "ถนน", "ตลาด",
                          "ทริป", "เดินทาง", "แนะนำ", "สวย", "ดัง", "ฮิต", "ถูก"}
        ):
            locations.append(t)

    return list(dict.fromkeys(locations))  # dedupe, preserve order


def _rank_suggest_words(
    hashtags: list[str],
    clean_tokens: list[str],
    sticker_texts: list[str],
) -> list[str]:
    """
    Rank keywords that are most likely to identify a specific place.
    Priority: hashtags > sticker OCR texts > description tokens.
    Filters out generic/noise tokens.
    """
    GENERIC_HASHTAGS = {
        "fyp", "foryou", "foryoupage", "xyzbca", "viral", "trending",
        "tiktok", "tiktokthailand", "tiktokviral", "fypシ",
        "แนะนำ", "รีวิว", "พากิน", "ฟีด", "เที่ยว", "กิน",
        "tiktokพากิน", "tiktoktravel", "อร่อยบอกต่อ",
        "ที่เที่ยว", "ที่กิน", "เช็คอิน",
    }

    scored: dict[str, float] = {}

    # Hashtags: highest priority (weight 3.0)
    for h in hashtags:
        h_lower = h.lower().strip()
        if h_lower and h_lower not in GENERIC_HASHTAGS and len(h_lower) >= 2:
            scored[h] = scored.get(h, 0) + 3.0

    # Sticker texts: high priority (weight 2.5) — often contain place names
    for st in sticker_texts:
        st_clean = st.strip()
        if st_clean and len(st_clean) >= 2:
            scored[st_clean] = scored.get(st_clean, 0) + 2.5

    # Clean description tokens: base priority (weight 1.0)
    token_freq = Counter(clean_tokens)
    for tok, freq in token_freq.items():
        if len(tok) >= 2:
            scored[tok] = scored.get(tok, 0) + 1.0 * freq

    # Sort by score descending → return top keywords
    ranked = sorted(scored, key=scored.get, reverse=True)
    return ranked[:30]  # top-30 suggest words


def extract_single(doc: dict[str, Any]) -> ExtractedContent:
    """
    Extract structured content from a single MongoDB content_metadata document.

    Args:
        doc: raw MongoDB document with schema:
             {video_id, metadata.video_metadata.{description, hashtags, author_id, ...}}

    Returns:
        ExtractedContent with all NLP-derived fields populated.
    """
    vm = doc.get("metadata", {}).get("video_metadata", {})
    video_id = str(doc.get("video_id", ""))
    author_id = str(vm.get("author_id", ""))
    description = vm.get("description", "") or ""
    raw_hashtags = vm.get("hashtags", []) or []

    # Sticker texts
    sticker_texts: list[str] = []
    for s in vm.get("stickers_on_item", []) or []:
        for txt in s.get("stickerText", []) or []:
            if txt:
                sticker_texts.append(str(txt))

    # Hashtags from both field + description text
    text_hashtags = _extract_hashtags_from_text(description)
    all_hashtags = list(dict.fromkeys(raw_hashtags + text_hashtags))  # dedupe

    # Thai segmentation on full text
    full_text = " ".join([description] + sticker_texts)
    tokens = _segment_thai(full_text)
    clean_tokens = [t for t in tokens if t not in THAI_STOPWORDS]

    # Location extraction
    location_tokens = _extract_location_tokens(tokens, full_text)

    # Suggest words
    suggest_words = _rank_suggest_words(all_hashtags, clean_tokens, sticker_texts)

    return ExtractedContent(
        video_id=video_id,
        author_id=author_id,
        description=description,
        hashtags=all_hashtags,
        sticker_texts=sticker_texts,
        tokens=tokens,
        clean_tokens=clean_tokens,
        suggest_words=suggest_words,
        location_tokens=location_tokens,
        playcount=vm.get("playcount", 0) or 0,
        diggcount=vm.get("diggcount", 0) or 0,
        sharecount=vm.get("sharecount", 0) or 0,
        collectcount=vm.get("collectcount", 0) or 0,
    )


def extract_batch_from_mongo(
    video_ids: list[str] | None = None,
    exclude_labeled: bool = True,
    batch_size: int = 5000,
) -> list[ExtractedContent]:
    """
    Extract content from all (or specified) videos in MongoDB.

    Args:
        video_ids:       specific video_ids to process (None = all)
        exclude_labeled: if True, skip videos already in place_videos (labeled data)
        batch_size:      MongoDB query batch size

    Returns:
        List of ExtractedContent objects.
    """
    from pymongo import MongoClient
    from tqdm import tqdm

    client = MongoClient(MONGO_URI)
    db = client[MONGO_TIKTOK_DB]

    # Get labeled video_ids to exclude
    labeled_vids: set[str] = set()
    if exclude_labeled:
        for doc in db.place_videos.find({}, {"video_ids": 1}):
            for vid in doc.get("video_ids", []):
                labeled_vids.add(str(vid))
        logger.info(f"  Excluding {len(labeled_vids)} already-labeled videos")

    # Determine which videos to process
    if video_ids is not None:
        target_ids = [v for v in video_ids if v not in labeled_vids]
    else:
        # Fetch all video_ids then filter
        all_ids = [str(d["video_id"]) for d in db.content_metadata.find({}, {"video_id": 1})]
        target_ids = [v for v in all_ids if v not in labeled_vids]

    logger.info(f"  Processing {len(target_ids)} unlabeled videos...")

    results: list[ExtractedContent] = []

    for i in tqdm(range(0, len(target_ids), batch_size), desc="extracting"):
        batch = target_ids[i:i + batch_size]
        cursor = db.content_metadata.find(
            {"video_id": {"$in": batch}},
            {
                "video_id": 1,
                "metadata.video_metadata": 1,
            },
        )
        for doc in cursor:
            try:
                ec = extract_single(doc)
                results.append(ec)
            except Exception as e:
                logger.warning(f"  Error extracting {doc.get('video_id')}: {e}")

    client.close()
    logger.info(f"  Extracted content for {len(results)} videos")
    return results
