#!/usr/bin/env python3
"""Quick analysis of entity resolution quality for filtering."""
import json, sys, os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

with open("data/entity_resolution_report.json") as f:
    report = json.load(f)

matches = report["matches"]
print(f"Total matches: {len(matches)}")

high = [m for m in matches if m["confidence"] == "high"]
medium = [m for m in matches if m["confidence"] == "medium"]
spatial = [m for m in matches if m.get("is_spatially_blocked")]
high_spatial = [m for m in matches if m["confidence"] == "high" and m.get("is_spatially_blocked")]

print(f"High confidence: {len(high)}")
print(f"Medium confidence: {len(medium)}")
print(f"Spatially blocked: {len(spatial)}")
print(f"High + Spatially blocked: {len(high_spatial)}")

# Token length analysis
tokens = [m["matched_token"] for m in matches]
short = [t for t in tokens if len(t) <= 3]
print(f"\nShort tokens (<=3 chars): {len(short)}")
print(f"Top short tokens: {Counter(short).most_common(20)}")

len_dist = defaultdict(int)
for t in tokens:
    len_dist[len(t)] += 1
print("\nToken length distribution:")
for l in sorted(len_dist.keys())[:12]:
    print(f"  len={l}: {len_dist[l]} matches")

# Clean filter: high + spatial + token>3
clean = [m for m in matches
         if m["confidence"] == "high"
         and m.get("is_spatially_blocked")
         and len(m["matched_token"]) > 3]
clean_places = set(m["place_id"] for m in clean)
print(f"\nAfter filter (high + spatial + token>3):")
print(f"  Matches: {len(clean)}")
print(f"  Unique places: {len(clean_places)}")

# What gets removed
removed = [m for m in matches
           if not (m["confidence"] == "high"
                   and m.get("is_spatially_blocked")
                   and len(m["matched_token"]) > 3)]
print(f"\nRemoved: {len(removed)} matches")
rem_tokens = Counter(m["matched_token"] for m in removed).most_common(25)
print(f"Top removed tokens: {rem_tokens}")
