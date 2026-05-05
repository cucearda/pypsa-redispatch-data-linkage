# Cluster Match Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `cluster_match.py` — a standalone second-pass script that reads the existing lookup CSV, finds aggregated Cluster entries, strips noise from names, runs fuzzy + LLM matching, and writes a list of matched `plant_ids` back to the CSV.

**Architecture:** Separate script (does not modify `plant_match_pre_filtering.py`). Imports shared helpers (load_plants, normalise_plant_name, PowerPlant, FUEL_FILTER, etc.) from the main pipeline module. Adds a `plant_ids` column to the existing lookup CSV and overwrites in-place after each LLM chunk.

**Tech Stack:** Python 3, pandas, rapidfuzz, anthropic SDK, pydantic — all already in `.venv/`

---

### Task 1: Noise stripping helpers + tests

**Files:**
- Create: `cluster_match.py`
- Create: `tests/test_cluster_strip.py`

**Step 1: Create the test file**

```python
# tests/test_cluster_strip.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re

CLUSTER_PREFIX_RE = re.compile(
    r"^[A-Za-z]+\s+(?:NWAK-)?Cluster\s+(?:EE\s+)?(?:\d+\s+)?",
    re.IGNORECASE,
)
TURBINE_SUFFIX_RE = re.compile(r"(\s+T\d+)+$", re.IGNORECASE)

def strip_cluster_noise(name: str) -> str:
    name = CLUSTER_PREFIX_RE.sub("", name)
    name = TURBINE_SUFFIX_RE.sub("", name)
    return name.strip()

def test_strip_shn_single_suffix():
    assert strip_cluster_noise("SHN Cluster Süderdonn T411") == "Süderdonn"

def test_strip_shn_multi_suffix():
    assert strip_cluster_noise("SHN Cluster Süderdonn T412 T413") == "Süderdonn"

def test_strip_ava():
    assert strip_cluster_noise("AVA Cluster Alfstedt") == "Alfstedt"

def test_strip_bag_nwak_with_number():
    assert strip_cluster_noise("BAG NWAK-Cluster 17 Altheim") == "Altheim"

def test_strip_ttg_with_ee():
    assert strip_cluster_noise("TTG Cluster EE Bechterdissen T411") == "Bechterdissen"

def test_strip_croc():
    assert strip_cluster_noise("CROC Cluster Görries") == "Görries"

def test_strip_no_suffix():
    assert strip_cluster_noise("AVA Cluster Helmstedt") == "Helmstedt"

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
```

**Step 2: Run tests — expect all FAIL (function not yet in cluster_match.py)**

```bash
cd /Users/a215482/Desktop/Uni/neu_thesis && .venv/bin/python3 tests/test_cluster_strip.py
```

(Will work since stripping is defined in the test file itself — confirms the regexes are correct before wiring them into the script.)

Expected: all `PASS` — these test the regex logic directly.

**Step 3: Create `cluster_match.py` with regexes and strip function**

```python
"""
cluster_match.py

Second-pass matching for Cluster aggregated entries in the plant lookup table.
Reads plant_lookup_total_v1.csv, strips cluster name noise, runs fuzzy + LLM,
writes plant_ids back. Non-cluster aggregated rows are untouched.

Run after plant_match_pre_filtering.py:
    .venv/bin/python3 cluster_match.py
"""

import csv
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
    FUEL_FILTER, TSO_BBOX, BBOX_TOLERANCE, MIN_CAPACITY, REFERENCE_DATE,
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
```

**Step 4: Re-run tests against the imported function**

Update `tests/test_cluster_strip.py` to import from `cluster_match`:

```python
from cluster_match import strip_cluster_noise
# (remove the local definitions of CLUSTER_PREFIX_RE, TURBINE_SUFFIX_RE, strip_cluster_noise)
```

```bash
.venv/bin/python3 tests/test_cluster_strip.py
```

Expected: all `PASS`

**Step 5: Commit**

```bash
git add cluster_match.py tests/test_cluster_strip.py
git commit -m "feat: cluster_match skeleton with noise stripping"
```

---

### Task 2: Data loading helpers

**Files:**
- Modify: `cluster_match.py` — add `load_redispatch_index()`

**Step 1: Add the function**

Append to `cluster_match.py` after `strip_cluster_noise`:

```python
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
```

**Step 2: Smoke-test in notebook or REPL**

```python
from cluster_match import load_redispatch_index
idx = load_redispatch_index("Redispatch_Daten.csv")
print(idx.get("SHN Cluster Süderdonn T411"))
# expected: ('Erneuerbar', 'TenneT DE') or similar
print(len(idx))
```

**Step 3: Commit**

```bash
git add cluster_match.py
git commit -m "feat: add load_redispatch_index to cluster_match"
```

---

### Task 3: LLM schema + prompt builder

**Files:**
- Modify: `cluster_match.py` — add Pydantic models and `build_cluster_prompt()`

**Step 1: Add Pydantic models**

```python
# ── Pydantic schema for cluster LLM output ────────────────────────────────────
class ClusterMatch(BaseModel):
    redispatch_name: str
    plant_ids:       list[int]   # empty list = no plausible match
    confidence:      Literal["high", "medium", "low", "none"]
    reasoning:       str

class ClusterMatchBatch(BaseModel):
    matches: list[ClusterMatch]
```

**Step 2: Add prompt builder**

```python
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
```

**Step 3: Verify schema serialises correctly**

```python
from cluster_match import ClusterMatchBatch
import json
print(json.dumps(ClusterMatchBatch.model_json_schema(), indent=2))
# Should show matches[] with plant_ids as array of integers
```

**Step 4: Commit**

```bash
git add cluster_match.py
git commit -m "feat: add ClusterMatchBatch schema and prompt builder"
```

---

### Task 4: Main pipeline — pre-filter + fuzzy stage

**Files:**
- Modify: `cluster_match.py` — add `main()` up to (not including) LLM calls

**Step 1: Add main() with loading + cluster detection**

```python
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
            and p.is_operational(REFERENCE_DATE)
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

if __name__ == "__main__":
    main()
```

**Step 2: Dry-run (no LLM yet) to verify pre-filter works**

```bash
.venv/bin/python3 cluster_match.py
```

Expected output: prints each cluster entry with its stripped query and candidate count. No API calls yet.

**Step 3: Commit**

```bash
git add cluster_match.py
git commit -m "feat: cluster_match pre-filter and fuzzy stage"
```

---

### Task 5: LLM stage + CSV write

**Files:**
- Modify: `cluster_match.py` — complete `main()` with LLM calls and CSV overwrite

**Step 1: Add LLM stage after the fuzzy loop**

Append inside `main()` after the fuzzy loop:

```python
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

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=[{
                "name": "submit_cluster_matches",
                "description": "Submit cluster match results for all entries",
                "input_schema": ClusterMatchBatch.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": "submit_cluster_matches"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        batch = ClusterMatchBatch.model_validate(response.content[0].input)

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
```

**Step 2: Run the full script**

```bash
.venv/bin/python3 cluster_match.py
```

Expected: processes ~60 cluster entries, makes 3 Claude API calls (60 / CHUNK_SIZE=20), overwrites `plant_lookup_total_v1.csv`.

**Step 3: Verify output**

```python
import pandas as pd
df = pd.read_csv("plant_lookup_total_v1.csv")
print(df[df["source"] == "cluster_match"][["redispatch_name", "plant_ids", "confidence"]].head(10))
print(df["source"].value_counts())
```

Expected: `plant_ids` column populated with comma-separated integers for matched clusters.

**Step 4: Commit**

```bash
git add cluster_match.py
git commit -m "feat: complete cluster_match LLM stage and CSV overwrite"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add cluster_match.py to run order and key files table**

In the **Run order** section, add:
```markdown
# Step 3 — match aggregated cluster entries
python cluster_match.py
```

In the **Key files** table, add:
```markdown
| `cluster_match.py` | Second-pass: matches Cluster aggregated entries to lists of plant_ids |
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add cluster_match.py to CLAUDE.md"
```
