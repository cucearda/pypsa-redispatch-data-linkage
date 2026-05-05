"""
Fuzzy-match redispatch plant names (BETROFFENE_ANLAGE) to germany_power_plants_names.csv.
Low-confidence matches are sent to Claude for disambiguation.

Outputs: plant_lookup.csv with columns:
  redispatch_name | matched_name | score | source | reasoning | needs_review
"""

import csv
import json
import os
from typing import List

import anthropic
from pydantic import BaseModel
from rapidfuzz import process, fuzz, utils

REDISPATCH       = "Redispatch_Daten.csv"
PLANTS           = "germany_power_plants_names_unique.csv"
OUTPUT           = "plant_lookup.csv"
REDISPATCH_SEP   = ";"
REDISPATCH_COL   = "BETROFFENE_ANLAGE"
REVIEW_THRESHOLD = 70   # fuzzy scores below this go to Claude
LLM_CANDIDATES   = 15   # top-N candidates to show Claude per item
CHUNK_SIZE       = 30   # items per Claude API call

# ── pydantic schema for structured output ─────────────────────────────────────
class PlantMatch(BaseModel):
    redispatch_name: str
    matched_name: str   # exact string from candidates, or "" if no good match
    reasoning: str      # brief explanation

class PlantMatches(BaseModel):
    matches: List[PlantMatch]

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")

with open(PLANTS, encoding="utf-8-sig") as f:
    plant_names = [row["Name"].strip()
                   for row in csv.DictReader(f)
                   if row["Name"].strip()]

with open(REDISPATCH, encoding="utf-8-sig") as f:
    redispatch_names = sorted({
        row[REDISPATCH_COL].strip()
        for row in csv.DictReader(f, delimiter=REDISPATCH_SEP)
        if row[REDISPATCH_COL].strip()
    })

print(f"  Redispatch names : {len(redispatch_names)}")
print(f"  Plant names      : {len(plant_names)}")

# ── fuzzy match ───────────────────────────────────────────────────────────────
print("Fuzzy matching …")

rows = []
review_items = []   # (index, redispatch_name, top_candidates)

for name in redispatch_names:
    results = process.extract(
        name, plant_names,
        scorer=fuzz.token_set_ratio,
        processor=utils.default_process,
        limit=LLM_CANDIDATES
    )
    top_score = results[0][1]
    # Among tied top scores prefer the most specific (most-token) candidate
    tied = [r for r in results if r[1] == top_score]
    best, score, _ = max(tied, key=lambda r: len(r[0].split()))
    top_candidates = [r[0] for r in results]

    row = {
        "redispatch_name": name,
        "matched_name":    best,
        "score":           round(score, 1),
        "source":          "fuzzy",
        "reasoning":       "",
        "needs_review":    "YES" if score < REVIEW_THRESHOLD else "",
        "_candidates":     top_candidates,   # internal, stripped before writing
    }
    rows.append(row)

    if score < REVIEW_THRESHOLD:
        review_items.append((len(rows) - 1, name, top_candidates))

high_conf = len(rows) - len(review_items)
print(f"  High confidence (≥{REVIEW_THRESHOLD}): {high_conf}")
print(f"  Sending to Claude              : {len(review_items)}")

fieldnames = ["redispatch_name", "matched_name", "score", "source", "reasoning", "needs_review"]
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in fieldnames})

print(f"Written → {OUTPUT}")

# # ── LLM disambiguation ────────────────────────────────────────────────────────
# if review_items:
#     client = anthropic.Anthropic()

#     plant_list_text = "\n".join(f"{i+1}. {n}" for i, n in enumerate(plant_names))

#     system_blocks = [
#         {
#             "type": "text",
#             "text": (
#                 "You are an expert on German power plants and electricity grid infrastructure. "
#                 "Your task is to match plant names used in redispatch data to the correct name "
#                 "in a power plant database. The names refer to the same physical plants but use "
#                 "different conventions (abbreviations, operator names in brackets, block numbers, "
#                 "German vs abbreviated names, etc.).\n\n"
#                 "Rules:\n"
#                 "- matched_name MUST be copied exactly (character-for-character) from the database list below\n"
#                 "- If no plant in the database is a plausible match, set matched_name to \"\"\n"
#                 "- Keep reasoning brief (1-2 sentences)\n\n"
#                 "Complete plant database:\n"
#                 + plant_list_text
#             ),
#             "cache_control": {"type": "ephemeral"},   # cached after first call
#         }
#     ]

#     def build_user_prompt(chunk):
#         items_text = [
#             f"Item {idx}: \"{rd_name}\""
#             for idx, (_, rd_name, _) in enumerate(chunk, 1)
#         ]
#         return (
#             "Match each redispatch plant name to the best entry in the plant database.\n\n"
#             + "\n".join(items_text)
#         )

#     # smaller chunks — user message is tiny now, system is cached
#     chunks = [
#         review_items[i:i + CHUNK_SIZE]
#         for i in range(0, len(review_items), CHUNK_SIZE)
#     ]

#     for chunk_num, chunk in enumerate(chunks, 1):
#         print(f"  Claude call {chunk_num}/{len(chunks)} ({len(chunk)} items) …", flush=True)

#         with client.messages.stream(
#             model="claude-opus-4-6",
#             max_tokens=8192,
#             thinking={"type": "adaptive"},
#             system=system_blocks,
#             messages=[{"role": "user", "content": build_user_prompt(chunk)}],
#             output_config={
#                 "format": {
#                     "type": "json_schema",
#                     "schema": PlantMatches.model_json_schema(),
#                 }
#             },
#             betas=["context-1m-2025-08-07"],   # 1M context window for the full plant list
#         ) as stream:
#             final = stream.get_final_message()

#         text = next(b.text for b in final.content if b.type == "text")
#         result = PlantMatches.model_validate_json(text)

#         # merge back by position (Claude returns items in order)
#         for i, match in enumerate(result.matches):
#             row_idx, _, _ = chunk[i]
#             rows[row_idx]["matched_name"]  = match.matched_name
#             rows[row_idx]["source"]        = "claude" if match.matched_name else "unmatched"
#             rows[row_idx]["reasoning"]     = match.reasoning
#             rows[row_idx]["needs_review"]  = "" if match.matched_name else "YES"

# # ── write output ──────────────────────────────────────────────────────────────
# fieldnames = ["redispatch_name", "matched_name", "score",
#               "source", "reasoning", "needs_review"]

# with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     for row in rows:
#         writer.writerow({k: row[k] for k in fieldnames})

# # ── summary ───────────────────────────────────────────────────────────────────
# n_fuzzy     = sum(1 for r in rows if r["source"] == "fuzzy")
# n_claude    = sum(1 for r in rows if r["source"] == "claude")
# n_unmatched = sum(1 for r in rows if r["source"] == "unmatched")
# n_review    = sum(1 for r in rows if r["needs_review"])

# print(f"\nDone → {OUTPUT}")
# print(f"  Matched by fuzzy  : {n_fuzzy}")
# print(f"  Matched by Claude : {n_claude}")
# print(f"  Still unmatched   : {n_unmatched}")
# print(f"  Needs manual review: {n_review}")
