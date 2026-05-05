# Cluster Match Pipeline Design
**Date:** 2026-05-05

## Problem

~60 of 158 aggregated redispatch entries are Cluster-type entries (e.g. `SHN Cluster Süderdonn T411`, `AVA Cluster Altheim`). These represent groups of co-located plants controlled together by a DSO. The main pipeline skips them entirely. This script attempts to match each cluster to a list of plant_ids from the PyPSA database.

Non-cluster aggregated entries (`_CR_` control reserve nodes, `EE ...` regional buckets, `VNB`, `Börse`) are genuinely unmatchable and remain untouched.

## Approach

Separate script `cluster_match.py` — reads and overwrites the existing lookup CSV. The main pipeline (`plant_match_pre_filtering.py`) is unchanged.

## Inputs

| File | Purpose |
|---|---|
| `plant_lookup_total_v1.csv` | Source of cluster rows to process (`source=aggregated`, name contains `Cluster`) |
| `Redispatch_Daten.csv` | Joined on `BETROFFENE_ANLAGE` to recover `PRIMAERENERGIEART` + `ANWEISENDER_UENB` for pre-filtering |
| `powerplants_pypsa_germany_merged.csv` | Plant candidate database |

## Noise Stripping

Two regexes applied in sequence:

```
Prefix: ^[A-Za-z]+\s+(?:NWAK-)?Cluster\s+(?:EE\s+)?(?:\d+\s+)?
Suffix: (\s+T\d+)+$
```

Examples:
- `SHN Cluster Süderdonn T411 T412` → `Süderdonn`
- `BAG NWAK-Cluster 17 Altheim` → `Altheim`
- `TTG Cluster EE Bechterdissen T411` → `Bechterdissen`
- `CROC Cluster Görries` → `Görries`

## Candidate Pre-filtering

Same logic as main pipeline:
- Fuel type filter via `PRIMAERENERGIEART` → `FUEL_FILTER`
- TSO bounding box filter via `ANWEISENDER_UENB`
- Min capacity ≥ 1 MW (0.0 MW unknown kept)
- `is_operational(REFERENCE_DATE)`

## Fuzzy Matching

- Scorer: `fuzz.WRatio`
- Top **40** candidates passed to LLM (higher than main pipeline's 20 — clusters span more plants)
- Same `normalise_plant_name()` prefix stripping on candidate names

## LLM Stage

New Pydantic models (defined in `cluster_match.py`, not modifying main pipeline):

```python
class ClusterMatch(BaseModel):
    redispatch_name: str
    plant_ids:   list[int]   # empty list = no plausible match
    confidence:  Literal["high", "medium", "low", "none"]
    reasoning:   str

class ClusterMatchBatch(BaseModel):
    matches: list[ClusterMatch]
```

Prompt context: explain these are groups of co-located plants controlled together — return **all** candidate ids that plausibly belong to the cluster. Empty list if no candidate clearly belongs.

- Model: `claude-sonnet-4-6`
- `CHUNK_SIZE = 20`
- Incremental append after each chunk (crash-safe)

## Output

Overwrites `plant_lookup_total_v1.csv` in-place. Adds `plant_ids` column (blank for all non-cluster rows).

For matched cluster rows:

| Column | Value |
|---|---|
| `source` | `cluster_match` or `cluster_unmatched` |
| `plant_ids` | `"141,142,143"` or `""` |
| `confidence` | from LLM |
| `reasoning` | from LLM |
| `needs_review` | `""` if high/medium, `"YES"` if low/none/empty |

All other rows (individual plant matches, non-cluster aggregated) are written back unchanged with a blank `plant_ids` value.

## Tuning Constants

```python
CLUSTER_TOP_K  = 40   # candidates passed to LLM for cluster entries
CHUNK_SIZE     = 20   # cluster entries per Claude API call
```
