"""
cluster_match.py

Second-pass matching for Cluster aggregated entries in the plant lookup table.
Reads plant_lookup_total_v1.csv, strips cluster name noise, runs fuzzy + LLM,
writes plant_ids back. Non-cluster aggregated rows are untouched.

Run after plant_match_pre_filtering.py:
    .venv/bin/python3 cluster_match.py
"""

import re
import os
import dotenv
from typing import Literal

import pandas as pd
import anthropic
from pydantic import BaseModel
from rapidfuzz import process, fuzz, utils as fuzz_utils

from plant_match_pre_filtering import (
    load_plants,
    normalise_plant_name,
    FUEL_FILTER, TSO_BBOX, BBOX_TOLERANCE, MIN_CAPACITY,
    REDISPATCH_SEP, POWERPLANT_SEP,
)

dotenv.load_dotenv()

# ── files ─────────────────────────────────────────────────────────────────────
LOOKUP_FILE     = "plant_lookup_total_v1.csv"
REDISPATCH_FILE = "Redispatch_Daten.csv"
PLANTS_FILE     = "powerplants_pypsa_germany_merged.csv"

# ── tuning ────────────────────────────────────────────────────────────────────
CLUSTER_TOP_K = 40
CHUNK_SIZE    = 20

# ── noise stripping ───────────────────────────────────────────────────────────
CLUSTER_PREFIX_RE = re.compile(
    r"^[A-Za-z]+\s+(?:NWAK-)?Cluster\s+(?:EE\s+)?(?:\d+\s+)?",
    re.IGNORECASE,
)
TURBINE_SUFFIX_RE = re.compile(r"(\s+T\d+)+$", re.IGNORECASE)


def strip_cluster_noise(name: str) -> str:
    name = CLUSTER_PREFIX_RE.sub("", name)
    name = TURBINE_SUFFIX_RE.sub("", name)
    return name.strip()


def load_redispatch_index(path: str) -> dict[str, tuple[str, str]]:
    """Return {plant_name: (energy_type, tso)} — first occurrence per name."""
    df = pd.read_csv(path, encoding="utf-8-sig", sep=REDISPATCH_SEP)
    df.columns = df.columns.str.strip()
    for col in ("BETROFFENE_ANLAGE", "PRIMAERENERGIEART", "ANWEISENDER_UENB"):
        df[col] = df[col].str.strip()
    df = df.drop_duplicates(subset=["BETROFFENE_ANLAGE"])
    return {
        row["BETROFFENE_ANLAGE"]: (row["PRIMAERENERGIEART"], row["ANWEISENDER_UENB"])
        for _, row in df.iterrows()
    }


# ── Pydantic schema for cluster LLM output ────────────────────────────────────
class ClusterMatch(BaseModel):
    redispatch_name: str
    plant_ids:       list[int]   # empty list = no plausible match
    confidence:      Literal["high", "medium", "low", "none"]
    reasoning:       str

class ClusterMatchBatch(BaseModel):
    matches: list[ClusterMatch]


def build_cluster_prompt(chunk: list[tuple[str, str, list]]) -> str:
    """chunk: list of (redispatch_name, stripped_query, candidates)"""
    parts = []
    for i, (redispatch_name, query, candidates) in enumerate(chunk, 1):
        candidate_lines = "\n".join(f"    {c.summary()}" for c in candidates)
        if not candidate_lines:
            candidate_lines = "    (no candidates after pre-filtering)"
        parts.append(
            f"Entry {i}: \"{redispatch_name}\" (location query: \"{query}\")\n"
            f"  Candidates:\n{candidate_lines}"
        )
    return (
        "Each entry is a DSO-controlled cluster of co-located renewable plants.\n"
        "Return ALL candidate plant_ids that plausibly belong to the cluster.\n"
        "Use location, fuel type, and capacity to decide.\n"
        "Return an empty list if no candidate clearly belongs.\n\n"
        + "\n\n".join(parts)
    )


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading data …")
    plants      = load_plants(PLANTS_FILE)
    plant_by_id = {p.id: p for p in plants}
    rd_index    = load_redispatch_index(REDISPATCH_FILE)

    lookup_df = pd.read_csv(LOOKUP_FILE, encoding="utf-8")
    if "plant_ids" not in lookup_df.columns:
        lookup_df["plant_ids"] = ""

    cluster_mask = (
        (lookup_df["source"] == "aggregated") &
        lookup_df["redispatch_name"].str.contains(r"\bCluster\b", case=False, na=False)
    )
    cluster_rows = lookup_df[cluster_mask]
    print(f"  Cluster entries to process: {len(cluster_rows)}")

    llm_queue: list[tuple[int, str, str, list]] = []

    for df_idx, row in cluster_rows.iterrows():
        name = row["redispatch_name"]
        energy_type, tso = rd_index.get(name, ("Erneuerbar", ""))

        stripped = strip_cluster_noise(name)
        if not stripped:
            print(f"  [skip] {name!r} — nothing left after stripping")
            continue

        candidates = [
            p for p in plants
            if p.matches_fuel(energy_type)
            and p.in_tso_area(tso)
            and p.above_min_capacity()
        ]

        if not candidates:
            print(f"  [no candidates] {name!r}")
            lookup_df.at[df_idx, "source"]       = "cluster_unmatched"
            lookup_df.at[df_idx, "reasoning"]    = "No candidates after pre-filtering."
            lookup_df.at[df_idx, "needs_review"] = "YES"
            continue

        candidate_names = [normalise_plant_name(p.name) for p in candidates]
        results = process.extract(
            stripped,
            candidate_names,
            scorer=fuzz.WRatio,
            processor=fuzz_utils.default_process,
            limit=CLUSTER_TOP_K,
        )

        seen_ids: set[int] = set()
        top_k: list = []
        for _name, _score, idx in results:
            p = candidates[idx]
            if p.id not in seen_ids:
                seen_ids.add(p.id)
                top_k.append(p)

        print(f"  [→ llm] {name!r} → query={stripped!r} ({len(top_k)} candidates)")
        llm_queue.append((df_idx, name, stripped, top_k))

    print(f"  LLM queue: {len(llm_queue)} entries")

    if not llm_queue:
        print("No cluster entries to send to LLM.")
        lookup_df.to_csv(LOOKUP_FILE, index=False, encoding="utf-8")
        return

    print(f"\nSending {len(llm_queue)} cluster entries to Claude …")
    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    system_prompt = (
        "You are an expert on German power plants and electricity grid infrastructure. "
        "Each entry is a CLUSTER — a group of co-located renewable plants controlled "
        "together by a DSO for grid balancing. "
        "Return ALL plant_ids from the candidate list that plausibly belong to the cluster. "
        "Use location name, fuel type, and capacity to decide. "
        "Return an empty list if no candidate clearly belongs to this cluster."
    )

    chunks = [llm_queue[i:i+CHUNK_SIZE] for i in range(0, len(llm_queue), CHUNK_SIZE)]

    for chunk_num, chunk in enumerate(chunks, 1):
        print(f"  Claude call {chunk_num}/{len(chunks)} ({len(chunk)} items) …", flush=True)

        user_prompt = build_cluster_prompt(
            [(name, query, cands) for _, name, query, cands in chunk]
        )

        raw_input = None
        for attempt in range(1, 4):
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=system_prompt,
                tools=[{
                    "name": "submit_cluster_matches",
                    "description": "Submit cluster match results for all entries",
                    "input_schema": ClusterMatchBatch.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": "submit_cluster_matches"},
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_input = response.content[0].input if response.content else {}
            if raw_input.get("matches"):
                break
            print(f"  [warn] attempt {attempt}: empty/invalid response, retrying …", flush=True)
        else:
            print(f"  [error] chunk {chunk_num}: all retries failed, marking as unmatched")
            for df_idx, name, _, _ in chunk:
                lookup_df.at[df_idx, "source"]       = "cluster_unmatched"
                lookup_df.at[df_idx, "reasoning"]    = "LLM returned empty response after retries."
                lookup_df.at[df_idx, "needs_review"] = "YES"
                print(f"  [cluster_unmatched] {name!r}")
            lookup_df.to_csv(LOOKUP_FILE, index=False, encoding="utf-8")
            print(f"  Chunk {chunk_num} saved to {LOOKUP_FILE}", flush=True)
            continue

        batch = ClusterMatchBatch.model_validate(raw_input)

        for i, match in enumerate(batch.matches):
            df_idx, name, _, _ = chunk[i]
            ids = [pid for pid in match.plant_ids if pid in plant_by_id]

            if ids:
                lookup_df.at[df_idx, "source"]       = "cluster_match"
                lookup_df.at[df_idx, "plant_ids"]    = ",".join(str(pid) for pid in ids)
                lookup_df.at[df_idx, "confidence"]   = match.confidence
                lookup_df.at[df_idx, "reasoning"]    = match.reasoning
                lookup_df.at[df_idx, "needs_review"] = (
                    "" if match.confidence in ("high", "medium") else "YES"
                )
                print(f"  [cluster_match]   {name!r} → {ids} (conf={match.confidence})")
            else:
                lookup_df.at[df_idx, "source"]       = "cluster_unmatched"
                lookup_df.at[df_idx, "confidence"]   = match.confidence
                lookup_df.at[df_idx, "reasoning"]    = match.reasoning
                lookup_df.at[df_idx, "needs_review"] = "YES"
                print(f"  [cluster_unmatched] {name!r}")

        lookup_df.to_csv(LOOKUP_FILE, index=False, encoding="utf-8")
        print(f"  Chunk {chunk_num} saved to {LOOKUP_FILE}", flush=True)

    n_matched   = (lookup_df["source"] == "cluster_match").sum()
    n_unmatched = (lookup_df["source"] == "cluster_unmatched").sum()
    print(f"\nDone → {LOOKUP_FILE}")
    print(f"  Cluster matched  : {n_matched}")
    print(f"  Cluster unmatched: {n_unmatched}")


if __name__ == "__main__":
    main()
