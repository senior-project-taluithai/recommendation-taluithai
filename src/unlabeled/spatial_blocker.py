"""
Spatial Blocking — Narrow TAT place search space using location mentions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uses province / district names extracted from TikTok hashtags, sticker texts,
and description to reduce the candidate places for fuzzy matching.

Logic:
  1. Build a lookup of province & district names (Thai)
  2. For each video, scan hashtags + suggest_words + location_tokens
     for province/district name matches
  3. Return a reduced candidate list from tat.places
  4. If no location match → fall back to full candidate list

This dramatically reduces the fuzzy matching search space from
~20K places to typically hundreds in a single province.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg2

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import PG_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class LocationBlock:
    """Spatial block assignment for a video."""
    province_ids: list[int] = field(default_factory=list)
    district_ids: list[int] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    is_blocked: bool = False  # True if we narrowed the search space


# ─── Location gazetteer ───────────────────────────────────────────────────────
# Province name aliases (common abbreviations and variations)
PROVINCE_ALIASES: dict[str, str] = {
    "กทม": "กรุงเทพมหานคร",
    "กรุงเทพ": "กรุงเทพมหานคร",
    "กรุงเทพฯ": "กรุงเทพมหานคร",
    "เชียงใหม่": "เชียงใหม่",
    "เชียงราย": "เชียงราย",
    "ภูเก็ต": "ภูเก็ต",
    "พัทยา": "ชลบุรี",       # Pattaya is in Chonburi
    "หัวหิน": "ประจวบคีรีขันธ์",
    "เกาะสมุย": "สุราษฎร์ธานี",
    "เกาะช้าง": "ตราด",
    "เกาะลิเป๊ะ": "สตูล",
    "เกาะพะงัน": "สุราษฎร์ธานี",
    "เกาะเสม็ด": "ระยอง",
    "เขาใหญ่": "นครราชสีมา",
    "อยุธยา": "พระนครศรีอยุธยา",
    "เกาะลันตา": "กระบี่",
    "ปาย": "แม่ฮ่องสอน",
    "กระบี่": "กระบี่",
    "สมุย": "สุราษฎร์ธานี",
    "หาดใหญ่": "สงขลา",
    "นครศรี": "นครศรีธรรมราช",
    "โคราช": "นครราชสีมา",
    "ขอนแก่น": "ขอนแก่น",
    "เลย": "เลย",
    "น่าน": "น่าน",
    "แม่ฮ่องสอน": "แม่ฮ่องสอน",
    "ลำปาง": "ลำปาง",
    "ลำพูน": "ลำพูน",
    "สุโขทัย": "สุโขทัย",
    "กาญจนบุรี": "กาญจนบุรี",
    "ราชบุรี": "ราชบุรี",
    "ตราด": "ตราด",
    "จันทบุรี": "จันทบุรี",
    "ระยอง": "ระยอง",
    "ชลบุรี": "ชลบุรี",
    "สุราษฎร์ธานี": "สุราษฎร์ธานี",
    "พังงา": "พังงา",
    "สตูล": "สตูล",
    "ตรัง": "ตรัง",
    "สงขลา": "สงขลา",
    "นครพนม": "นครพนม",
    "อุดรธานี": "อุดรธานี",
    "เพชรบุรี": "เพชรบุรี",
    "สระบุรี": "สระบุรี",
    "นครปฐม": "นครปฐม",
    "สมุทรปราการ": "สมุทรปราการ",
    "นนทบุรี": "นนทบุรี",
    "ปทุมธานี": "ปทุมธานี",
}

# English place name aliases
PROVINCE_EN_ALIASES: dict[str, str] = {
    "bangkok": "กรุงเทพมหานคร",
    "chiangmai": "เชียงใหม่",
    "chiang mai": "เชียงใหม่",
    "chiangrai": "เชียงราย",
    "chiang rai": "เชียงราย",
    "phuket": "ภูเก็ต",
    "pattaya": "ชลบุรี",
    "krabi": "กระบี่",
    "kosamui": "สุราษฎร์ธานี",
    "ko samui": "สุราษฎร์ธานี",
    "koh samui": "สุราษฎร์ธานี",
    "huahin": "ประจวบคีรีขันธ์",
    "hua hin": "ประจวบคีรีขันธ์",
    "kanchanaburi": "กาญจนบุรี",
    "ayutthaya": "พระนครศรีอยุธยา",
    "khao yai": "นครราชสีมา",
    "pai": "แม่ฮ่องสอน",
    "nan": "น่าน",
    "loei": "เลย",
    "sukhothai": "สุโขทัย",
    "trat": "ตราด",
    "rayong": "ระยอง",
    "koh chang": "ตราด",
    "koh lipe": "สตูล",
    "koh lanta": "กระบี่",
    "koh phangan": "สุราษฎร์ธานี",
    "koh samet": "ระยอง",
    "hat yai": "สงขลา",
    "khon kaen": "ขอนแก่น",
    "udon thani": "อุดรธานี",
    "chonburi": "ชลบุรี",
}


class SpatialBlocker:
    """
    Builds a gazetteer from PostgreSQL province/district data
    and uses it to spatially block TikTok video → candidate places.
    """

    def __init__(self):
        self.province_name_to_id: dict[str, int] = {}
        self.district_name_to_ids: dict[str, list[int]] = {}  # name → [district_ids]
        self.district_to_province: dict[int, int] = {}
        self.province_places: dict[int, list[dict]] = {}  # province_id → [{place_id, name, ...}]
        self._loaded = False

    def load_gazetteer(self):
        """Load province & district names from PostgreSQL."""
        if self._loaded:
            return

        conn = psycopg2.connect(**PG_CONFIG)
        cur = conn.cursor()

        # Provinces
        cur.execute("SELECT province_id, name FROM tat.provinces")
        for pid, name in cur.fetchall():
            self.province_name_to_id[name.strip()] = pid
            # Also index without "จังหวัด" prefix
            stripped = name.strip().replace("จังหวัด", "").strip()
            if stripped != name.strip():
                self.province_name_to_id[stripped] = pid

        # Add aliases
        for alias, canonical in PROVINCE_ALIASES.items():
            if canonical in self.province_name_to_id:
                self.province_name_to_id[alias] = self.province_name_to_id[canonical]

        for alias_en, canonical in PROVINCE_EN_ALIASES.items():
            if canonical in self.province_name_to_id:
                self.province_name_to_id[alias_en] = self.province_name_to_id[canonical]

        # Districts
        cur.execute("SELECT district_id, province_id, name FROM tat.districts")
        for did, pid, name in cur.fetchall():
            name_clean = name.strip()
            self.district_name_to_ids.setdefault(name_clean, []).append(did)
            self.district_to_province[did] = pid
            # Also without อำเภอ prefix
            stripped = name_clean.replace("อำเภอ", "").replace("เขต", "").strip()
            if stripped != name_clean:
                self.district_name_to_ids.setdefault(stripped, []).append(did)

        # Load places grouped by province
        cur.execute("""
            SELECT place_id, name, province_id, district_id, latitude, longitude
            FROM tat.places
            WHERE province_id IS NOT NULL
        """)
        for row in cur.fetchall():
            pid = row[2]
            self.province_places.setdefault(pid, []).append({
                "place_id": row[0],
                "name": row[1] or "",
                "province_id": pid,
                "district_id": row[3],
                "latitude": float(row[4]) if row[4] else 0.0,
                "longitude": float(row[5]) if row[5] else 0.0,
            })

        conn.close()
        self._loaded = True

        logger.info(
            f"  Gazetteer loaded: {len(self.province_name_to_id)} province entries, "
            f"{len(self.district_name_to_ids)} district entries, "
            f"{sum(len(v) for v in self.province_places.values())} places"
        )

    def match_location(
        self,
        hashtags: list[str],
        suggest_words: list[str],
        location_tokens: list[str],
    ) -> LocationBlock:
        """
        Try to identify province/district from video content.

        Scans hashtags, suggest_words, and location_tokens against
        the province/district gazetteer.
        """
        self.load_gazetteer()

        block = LocationBlock()
        seen_provinces: set[int] = set()
        seen_districts: set[int] = set()

        # All candidate terms to check (priority order)
        all_terms: list[str] = []
        # Hashtags first (highest confidence location signal)
        all_terms.extend(hashtags)
        # Location tokens from NLP extraction
        all_terms.extend(location_tokens)
        # Suggest words
        all_terms.extend(suggest_words)

        for term in all_terms:
            t = term.strip().lower()
            if not t or len(t) < 2:
                continue

            # Check province match
            for prov_name, pid in self.province_name_to_id.items():
                if pid in seen_provinces:
                    continue
                # Exact match or term contains province name
                pn = prov_name.lower()
                if pn == t or pn in t or t in pn:
                    # Avoid partial false positives (e.g. "น่า" matching "น่าน")
                    if len(t) >= 2 and len(pn) >= 2:
                        seen_provinces.add(pid)
                        block.province_ids.append(pid)
                        block.matched_terms.append(f"province:{prov_name}")

            # Check district match (only longer terms to avoid false positives)
            if len(t) >= 3:
                for dist_name, dids in self.district_name_to_ids.items():
                    dn = dist_name.lower()
                    if dn == t or (len(dn) >= 4 and dn in t):
                        for did in dids:
                            if did not in seen_districts:
                                seen_districts.add(did)
                                block.district_ids.append(did)
                                block.matched_terms.append(f"district:{dist_name}")
                                # Also add the parent province
                                parent_pid = self.district_to_province.get(did)
                                if parent_pid and parent_pid not in seen_provinces:
                                    seen_provinces.add(parent_pid)
                                    block.province_ids.append(parent_pid)

        block.is_blocked = len(block.province_ids) > 0
        return block

    def get_candidate_places(
        self,
        block: LocationBlock,
        max_candidates: int = 2000,
    ) -> list[dict]:
        """
        Get candidate TAT places based on spatial blocking result.

        If blocked (province matched) → return places in those provinces.
        If not blocked → return ALL places (full search, limited by max).
        """
        self.load_gazetteer()

        if block.is_blocked:
            candidates = []
            for pid in block.province_ids:
                candidates.extend(self.province_places.get(pid, []))

            # If district_ids are matched, boost those places to front
            if block.district_ids:
                dist_set = set(block.district_ids)
                in_district = [p for p in candidates if p.get("district_id") in dist_set]
                not_in_district = [p for p in candidates if p.get("district_id") not in dist_set]
                candidates = in_district + not_in_district

            return candidates[:max_candidates]
        else:
            # No location match — return all places
            all_places = []
            for places in self.province_places.values():
                all_places.extend(places)
            return all_places[:max_candidates]
