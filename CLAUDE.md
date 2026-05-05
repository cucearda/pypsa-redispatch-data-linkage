# Thesis — Redispatch Plant Matching Pipeline

## Project purpose

Match German TSO redispatch entries (`BETROFFENE_ANLAGE`) to individual power plants in the PyPSA power plant database. Output is a lookup table used for thesis analysis of German redispatch events.

## Run order

```bash
# Step 1 — aggregate individual turbines into farm-level entries
python aggregate_plants.py

# Step 2 — run the three-stage matching pipeline
python plant_match_pre_filtering.py

# Step 3 — match aggregated cluster entries
python cluster_match.py
```

## Key files

| File | Role |
|---|---|
| `Redispatch_Daten.csv` | Full raw redispatch dataset (semicolon-delimited) |
| `redispatch_top100.csv` | Development subset (first 100 unique entries) |
| `powerplants_pypsa_germany.csv` | Raw PyPSA power plant database (comma-delimited) |
| `powerplants_pypsa_germany_merged.csv` | Output of `aggregate_plants.py` — turbines merged into farms |
| `plant_lookup_first100_v2.csv` | Current matching output |
| `aggregate_plants.py` | Preprocessing: merges same-name co-located turbines into single farm entries |
| `plant_match_pre_filtering.py` | Main pipeline: rule filter → fuzzy → Claude LLM |
| `cluster_match.py` | Second-pass: matches Cluster aggregated entries to lists of plant_ids |

## Pipeline architecture (`plant_match_pre_filtering.py`)

### Stage 1 — Aggregated entry detection
Entries matching `AGGREGATED_RE` are skipped immediately (cannot map to a single plant):
- `_CR_` — control reserve nodes (e.g. `50H_MNS_CR_WIND`)
- `VNB` — DSO aggregation buckets
- `Cluster` — wind/PV clusters (e.g. `SHN Cluster Süderdonn`)
- `EE` prefix/suffix — regional renewable buckets
- `UW` — Umspannwerk (substation/grid node, aggregates all plants feeding into it)
- `Börse`, `Notfall-RD` — market/emergency virtual entries

### Stage 1 — Pre-filtering of candidates
For non-aggregated entries, the full German plant list is narrowed by:
1. **Fuel type** — `PRIMAERENERGIEART` → `FUEL_FILTER` (Konventionell / Erneuerbar / Sonstiges)
2. **TSO bounding box** — loose geographic rectangle per TSO with `BBOX_TOLERANCE = 0.8°`; plants with no coordinates are kept
3. **Min capacity** — drops plants < 1 MW; keeps 0.0 MW (unknown)

### Stage 2 — Fuzzy matching
- Scorer: `fuzz.WRatio` (handles partial matches; better than `token_set_ratio` for names like `"Wedel"` within `"Hamburg Wedel1"`)
- Query normalised before matching: TSO prefix (`50H `) and parenthesised noise (`(KapRes)`) stripped
- Top 20 candidates by score passed forward
- **Auto-accept** conditions (all three must hold):
  - `top_score >= AUTO_THRESHOLD (90)`
  - Only 1 candidate within `TIE_TOLERANCE (5)` points of top score
  - `len(matched_name) / len(query) >= 0.5` — guards against short substring false positives

### Stage 3 — Claude LLM disambiguation
- Ambiguous entries (tied or below threshold) sent to `claude-sonnet-4-6` in batches of 20
- Claude receives: plant name, energy type, TSO, and top-20 candidates with name/fueltype/capacity/coords
- Returns structured `PlantMatchBatch` via tool use — `plant_id` (integer) not name, to avoid same-name ambiguity
- API key read from `CLAUDE_API_KEY` env var (`.env` file supported via `dotenv`)

### Output written incrementally
- Pre-LLM results (fuzzy/aggregated/unmatched) written to `OUTPUT_FILE` before Claude stage
- Each Claude batch appended immediately after processing — crash-safe for long runs

## `aggregate_plants.py` details

Groups plants by exact name. Merges a group if all members fall within `MAX_FARM_RADIUS_KM = 50 km` of the group centroid. Merged entry gets:
- `capacity_mw` = sum of all turbines
- `lat/lon` = centroid
- `turbine_count` = number of turbines
- `source_ids` = comma-separated original ids

Groups spanning > 50 km are kept separate (different plants sharing a name, e.g. `Zweite Anlage`).

## German grid terminology

| Term | Meaning |
|---|---|
| `BETROFFENE_ANLAGE` | Affected plant (the redispatch target) |
| `PRIMAERENERGIEART` | Primary energy type: Konventionell / Erneuerbar / Sonstiges |
| `ANWEISENDER_UENB` | Instructing TSO (50Hertz / TenneT DE / Amprion / TransnetBW) |
| `UW` | Umspannwerk — substation/transformer station (aggregated node) |
| `OWP` | Offshore-Windpark — offshore wind farm (real matchable plant) |
| `SHN` | Schleswig-Holstein Netz — DSO for Schleswig-Holstein |
| `T412`, `T413` | Individual turbine numbers within a cluster |
| `CR_` | Control reserve node |
| `VNB` | Verteilernetzbetreiber — distribution grid operator (DSO) |

## Tuning constants

All in `plant_match_pre_filtering.py`:

```python
TOP_K          = 20    # candidates passed to LLM
AUTO_THRESHOLD = 90    # min fuzzy score for auto-accept
TIE_TOLERANCE  = 5     # score band for tie detection
MIN_CAPACITY   = 1.0   # MW floor for candidate plants
CHUNK_SIZE     = 20    # redispatch entries per Claude API call
BBOX_TOLERANCE = 0.8   # degrees padding on TSO bounding boxes
```

## Environment

- Python venv at `.venv/` — always run with `.venv/bin/python3`
- Dependencies: `anthropic`, `pandas`, `rapidfuzz`, `pydantic`, `python-dotenv`
- `CLAUDE_API_KEY` must be set in `.env`
- All input/output files are relative to the script directory; scripts `chdir` on startup
