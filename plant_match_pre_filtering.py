"""
plant_match_pre_filtering.py

Three-stage pipeline to match redispatch BETROFFENE_ANLAGE names to powerplants.csv:

  Stage 1 — Rule-based pre-filtering
    • Detect and skip aggregated/virtual entries (VNB nodes, Cluster, CR_ nodes, etc.)
    • Filter powerplants by fuel type (Konventionell / Erneuerbar / Sonstiges)
    • Filter powerplants by TSO geographic bounding box

  Stage 2 — Fuzzy top-k on filtered candidate set
    • Strip TSO prefixes (e.g. "50H ") and suffix noise ("(KapRes)") before matching
    • Top-k candidates always sent to LLM — no auto-accept

  Stage 3 — LLM disambiguation
    • Send ambiguous entries to Claude with full structured context:
      plant name, energy type, TSO, and top-k candidates (name, fueltype, capacity, coords)
    • Claude returns plant_id (integer), not just a name, to avoid same-name ambiguity

Output: plant_lookup_v2.csv
  redispatch_name | plant_id | matched_name | fueltype | capacity_mw
  | confidence | source | reasoning | needs_review
"""

import csv
import re
import os
import dotenv
from dataclasses import dataclass
from datetime import date
from typing import Optional, Literal

import pandas as pd

import anthropic
from pydantic import BaseModel
from rapidfuzz import process, fuzz, utils as fuzz_utils

dotenv.load_dotenv()

# ── files ─────────────────────────────────────────────────────────────────────
REDISPATCH_FILE = "Redispatch_Daten.csv"
PLANTS_FILE     = "powerplants_pypsa_germany_merged.csv"
OUTPUT_FILE     = "plant_lookup_total_v1.csv"
POWERPLANT_SEP = ","
REDISPATCH_SEP = ";"

# ── tuning ────────────────────────────────────────────────────────────────────
TOP_K           = 20              # fuzzy candidates passed to LLM
MIN_CAPACITY    = 1.0             # MW — skip plants too small for TSO redispatch (0.0 = unknown, kept)
CHUNK_SIZE      = 20              # redispatch entries per Claude API call

# ── TSO geographic bounding boxes [lat_min, lat_max, lon_min, lon_max] ────────
# These are intentionally loose — TSO boundaries are irregular.
# The filter reduces candidates; it doesn't need to be exact.
TSO_BBOX: dict[str, tuple[float, float, float, float]] = {
    "50Hertz":    (51.0, 55.5,  9.5, 15.5),  # East Germany + Hamburg/Berlin
    "TenneT DE":  (47.0, 55.5,  6.0, 15.5),  # North + South (complex, use wide box)
    "Amprion":    (49.0, 52.5,  5.5, 10.5),  # West Germany (NRW, Rhineland)
    "TransnetBW": (47.3, 49.8,  7.3, 10.6),  # Baden-Württemberg
}
BBOX_TOLERANCE = 0.8   # degrees padding at TSO boundaries

# ── fuel type mapping: PRIMAERENERGIEART → allowed powerplants Fueltype ───────
# Konventionell includes Hydro because pumped storage is used for grid balancing.
# Biogas/Biomass/Geothermal appear in both categories — do not exclude them.
FUEL_FILTER: dict[str, Optional[set[str]]] = {
    "Konventionell": {
        "Natural Gas", "Hard Coal", "Lignite", "Nuclear", "Oil",
        "Waste", "Biogas", "Solid Biomass", "Geothermal", "Other", "Hydro",
    },
    "Erneuerbar": {
        "Solar", "Wind", "Hydro", "Biogas", "Solid Biomass", "Geothermal",
    },
    "Sonstiges": None,  # no fuel filter — could be storage, cross-border, etc.
}

# ── patterns that identify non-individual-plant entries ───────────────────────
# These cannot be matched to a single row in powerplants.csv.
AGGREGATED_RE = re.compile(
    r"_CR_"                   # control reserve node: 50H_MNS_CR_KLM, AMP_..._CR_WIND
    r"|VNB\b"                 # DSO aggregation bucket
    r"|\bCluster\b"           # wind/PV cluster: AVA Cluster, SHN Cluster
    r"|\s+EE$"                # regional renewable bucket: "EE Bayern", "Baden-Württemberg Nord EE"
    r"|^EE\s"                 # starts with EE: "EE Hessen"
    r"|^Börse$"               # electricity market entry (not a plant)
    r"|^Notfall-RD"           # emergency virtual entries
    r"|^BETROFFENE_ANLAGE$",  # stray CSV header row
    re.IGNORECASE,
)

# ── TSO prefix to strip before fuzzy matching (redispatch query names) ────────
TSO_PREFIX_RE = re.compile(r"^50H\s+", re.IGNORECASE)

# ── parenthesised suffix noise to strip ──────────────────────────────────────
SUFFIX_NOISE_RE = re.compile(r"\s*\((?:KapRes|bnBm|SysRel|kurativ|[A-Za-z\s]+)\)\s*$")

# ── technical type prefixes to strip from powerplant candidate names ──────────
# Strips one or more consecutive type abbreviations so "Ready Gtkw Lippendorf"
# scores the same as "Lippendorf" against a query like "Lippendorf VE".
PLANT_PREFIX_RE = re.compile(
    r"^(?:"
    r"Ready|Bhkw|Gtkw|Gud|Hkw|Psw|Kng|Wp|Windpark|Windkraftanlage|Wka|Pv|Kw"
    r"|Pss|Ms|Ro" # Suffixes
    r")\s+",
    re.IGNORECASE,
)


# ── data structures ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RedispatchKey:
    plant_name:      str
    energy_type:     str   # Konventionell | Erneuerbar | Sonstiges
    instructing_tso: str


@dataclass
class PowerPlant:
    id:          int
    name:        str
    fueltype:    str
    technology:  str
    set_:        str
    capacity_mw: float
    lat:         Optional[float]
    lon:         Optional[float]
    date_in:     Optional[date]
    date_out:    Optional[date]

    def matches_fuel(self, energy_type: str) -> bool:
        allowed = FUEL_FILTER.get(energy_type)
        return True if allowed is None else self.fueltype in allowed

    def above_min_capacity(self) -> bool:
        return self.capacity_mw == 0.0 or self.capacity_mw >= MIN_CAPACITY


    # Currently not super usesful, but if using older dispatch dates, this could help filter out plants not yet commissioned at that time.
    # def is_operational(self, on: date) -> bool:
    #     if self.date_in is not None and self.date_in > on:
    #         return False   # not yet commissioned
    #     return True

    def in_tso_area(self, tso: str) -> bool:
        if self.lat is None or self.lon is None:
            return True   # no coordinates → don't filter out
        bbox = TSO_BBOX.get(tso)
        if bbox is None:
            return True   # unknown TSO → don't filter
        lat_min, lat_max, lon_min, lon_max = bbox
        t = BBOX_TOLERANCE
        return (lat_min - t <= self.lat <= lat_max + t and
                lon_min - t <= self.lon <= lon_max + t)

    def summary(self) -> str:
        loc = f"{self.lat:.3f}°N {self.lon:.3f}°E" if self.lat and self.lon else "no coords"
        tech = f" {self.technology}" if self.technology else ""
        return (f"id={self.id:<6}  {self.name:<55}  |  "
                f"{self.fueltype}{tech:<30}  |  "
                f"{self.capacity_mw:>8.1f} MW  |  {loc}")


# ── Pydantic schema for Claude structured output ──────────────────────────────
class PlantMatch(BaseModel):
    redispatch_name: str
    plant_id:        Optional[int]  # None = no plausible match in candidates
    confidence:      Literal["high", "medium", "low", "none"]
    reasoning:       str            # 1-2 sentences

class PlantMatchBatch(BaseModel):
    matches: list[PlantMatch]


# ── helpers ───────────────────────────────────────────────────────────────────
def normalise_for_matching(name: str) -> str:
    name = TSO_PREFIX_RE.sub("", name)
    name = SUFFIX_NOISE_RE.sub("", name)
    return name.strip()


def normalise_plant_name(name: str) -> str:
    """Strip leading technical type prefixes from powerplant candidate names."""
    prev = None
    while prev != name:
        prev = name
        name = PLANT_PREFIX_RE.sub("", name)
    return name.strip()


def load_plants(path: str) -> list[PowerPlant]:
    df = pd.read_csv(path, encoding="utf-8-sig", sep=POWERPLANT_SEP)
    print(f"  [load_plants] {len(df)} total rows read from {path}")
    df = df[df["Country"].str.strip() == "Germany"].copy()
    print(f"  [load_plants] {len(df)} rows after Germany filter")

    df["lat"]     = pd.to_numeric(df.get("lat"),     errors="coerce")
    df["lon"]     = pd.to_numeric(df.get("lon"),     errors="coerce")
    df["Capacity"]= pd.to_numeric(df.get("Capacity"),errors="coerce").fillna(0.0)
    df["DateIn"]  = pd.to_numeric(df.get("DateIn"),  errors="coerce")
    df["DateOut"] = pd.to_numeric(df.get("DateOut"), errors="coerce")
    df["Name"]    = df["Name"].str.strip()
    for col in ("Fueltype", "Technology", "Set"):
        df[col] = df.get(col, pd.Series("", index=df.index)).fillna("").str.strip()

    return [
        PowerPlant(
            id=int(row["id"]),
            name=row["Name"],
            fueltype=row["Fueltype"],
            technology=row["Technology"],
            set_=row["Set"],
            capacity_mw=float(row["Capacity"]),
            lat=None if pd.isna(row["lat"]) else float(row["lat"]),
            lon=None if pd.isna(row["lon"]) else float(row["lon"]),
            date_in=None  if pd.isna(row["DateIn"])  else date(int(row["DateIn"]),  1,  1),
            date_out=None if pd.isna(row["DateOut"]) else date(int(row["DateOut"]), 12, 31),
        )
        for _, row in df.iterrows()
    ]


def load_redispatch_keys(path: str) -> list[RedispatchKey]:
    df = pd.read_csv(path, encoding="utf-8-sig", sep=REDISPATCH_SEP)
    df.columns = df.columns.str.strip()
    print(f"  [load_redispatch] {len(df)} rows read from {path}")
    df["BETROFFENE_ANLAGE"] = df["BETROFFENE_ANLAGE"].str.strip()
    df["PRIMAERENERGIEART"] = df["PRIMAERENERGIEART"].str.strip()
    df["ANWEISENDER_UENB"]  = df["ANWEISENDER_UENB"].str.strip()

    df = df[df["BETROFFENE_ANLAGE"].notna() & (df["BETROFFENE_ANLAGE"] != "")]
    df = df.drop_duplicates(subset=["BETROFFENE_ANLAGE", "PRIMAERENERGIEART", "ANWEISENDER_UENB"])
    print(f"  [load_redispatch] {len(df)} unique entries after dedup")

    return [
        RedispatchKey(row["BETROFFENE_ANLAGE"], row["PRIMAERENERGIEART"], row["ANWEISENDER_UENB"])
        for _, row in df.iterrows()
    ]


def build_llm_prompt(chunk: list[tuple[RedispatchKey, list[PowerPlant]]]) -> str:
    parts = []
    for i, (key, candidates) in enumerate(chunk, 1):
        candidate_lines = "\n".join(f"    {c.summary()}" for c in candidates)
        if not candidate_lines:
            candidate_lines = "    (no candidates after pre-filtering)"
        parts.append(
            f"Entry {i}: \"{key.plant_name}\"\n"
            f"  Energy type : {key.energy_type}\n"
            f"  Instructing TSO: {key.instructing_tso}\n"
            f"  Candidates:\n{candidate_lines}"
        )
    return (
        "Match each redispatch plant name to the correct power plant from its candidate list.\n"
        "Return one result per entry, in the same order.\n\n"
        + "\n\n".join(parts)
    )


# ── main pipeline ─────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data …")
    plants = load_plants(PLANTS_FILE)
    keys   = load_redispatch_keys(REDISPATCH_FILE)
    print(f"  German plants    : {len(plants)}")
    print(f"  Unique RD entries: {len(keys)}")

    plant_by_id = {p.id: p for p in plants}

    # ── stage 1+2: pre-filter + fuzzy ─────────────────────────────────────────
    print("\nPre-filtering + fuzzy matching …")

    output_rows:   list[dict] = []
    llm_queue:     list[tuple[int, RedispatchKey, list[PowerPlant]]] = []  # (row_idx, key, candidates)

    for key in keys:
        row = {
            "redispatch_name": key.plant_name,
            "plant_id":        "",
            "matched_name":    "",
            "fueltype":        "",
            "capacity_mw":     "",
            "confidence":      "",
            "source":          "",
            "reasoning":       "",
            "needs_review":    "",
        }

        # ── skip aggregated / virtual entries ──────────────────────────────
        if AGGREGATED_RE.search(key.plant_name):
            print(f"  [aggregated]  {key.plant_name!r}")
            row["source"]      = "aggregated"
            row["reasoning"]   = "Aggregated node or virtual entry — not a single plant."
            output_rows.append(row)
            continue

        # ── pre-filter candidates ──────────────────────────────────────────
        candidates = [
            p for p in plants
            if p.matches_fuel(key.energy_type)
            and p.in_tso_area(key.instructing_tso)
            and p.above_min_capacity()
        ]

        # ── fuzzy match on normalised names ────────────────────────────────
        norm_query      = normalise_for_matching(key.plant_name)
        candidate_names = [normalise_plant_name(p.name) for p in candidates]

        if not candidates:
            print(f"  [no candidates] {key.plant_name!r}  (type={key.energy_type}, tso={key.instructing_tso})")
            row["source"]      = "unmatched"
            row["reasoning"]   = "No candidates after pre-filtering."
            row["needs_review"] = "YES"
            output_rows.append(row)
            continue

        results = process.extract(
            norm_query,
            candidate_names,
            scorer=fuzz.WRatio,
            processor=fuzz_utils.default_process,
            limit=TOP_K,
        )

        # deduplicate by plant id
        seen_ids: set[int] = set()
        top_k_plants: list[PowerPlant] = []
        for _name, _score, idx in results:
            p = candidates[idx]
            if p.id not in seen_ids:
                seen_ids.add(p.id)
                top_k_plants.append(p)

        print(f"  [→ llm]  {key.plant_name!r}  (candidates={len(top_k_plants)})")
        llm_queue.append((len(output_rows), key, top_k_plants))
        row["source"] = "pending_llm"

        output_rows.append(row)

    aggregated = sum(1 for r in output_rows if r["source"] == "aggregated")
    print(f"  Aggregated/skipped: {aggregated}")
    print(f"  Sending to Claude : {len(llm_queue)}")

    # ── write pre-LLM results ─────────────────────────────────────────────────
    fieldnames = [
        "redispatch_name", "plant_id", "matched_name", "fueltype",
        "capacity_mw", "confidence", "source", "reasoning", "needs_review",
    ]
    pre_llm_rows = [r for r in output_rows if r["source"] != "pending_llm"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pre_llm_rows)
    print(f"\nPre-LLM results written to {OUTPUT_FILE} ({len(pre_llm_rows)} rows)")

    # ── stage 3: LLM disambiguation ───────────────────────────────────────────
    if llm_queue:
        client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

        system_prompt = (
            "You are an expert on German power plants and electricity grid infrastructure. "
            "Match redispatch plant names (from TSO reports) to the correct entry in a "
            "power plant database. Names in redispatch data use TSO-internal shorthands, "
            "operator names, block numbers, or abbreviated city names that differ from the "
            "official MASTR database names.\n\n"
            "Rules:\n"
            "- plant_id MUST be an integer from the provided candidate list, or null if "
            "no candidate is a plausible match\n"
            "- Consider fuel type, capacity, and location alongside the name\n"
            "- Keep reasoning to 1-2 sentences\n"
            "- For block-level entries (e.g. 'Block A') match to the plant entry that "
            "covers that block if one exists"
        )

        chunks = [
            llm_queue[i : i + CHUNK_SIZE]
            for i in range(0, len(llm_queue), CHUNK_SIZE)
        ]

        for chunk_num, chunk in enumerate(chunks, 1):
            print(f"  Claude call {chunk_num}/{len(chunks)} ({len(chunk)} items) …", flush=True)

            user_prompt = build_llm_prompt([(key, cands) for _, key, cands in chunk])

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=system_prompt,
                tools=[{
                    "name": "submit_matches",
                    "description": "Submit the match result for every entry",
                    "input_schema": PlantMatchBatch.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": "submit_matches"},
                messages=[{"role": "user", "content": user_prompt}],
            )

            batch: PlantMatchBatch = PlantMatchBatch.model_validate(
                response.content[0].input
            )

            for i, match in enumerate(batch.matches):
                row_idx, key, _ = chunk[i]
                plant = plant_by_id.get(match.plant_id) if match.plant_id else None

                output_rows[row_idx]["source"]      = "claude" if plant else "unmatched"
                output_rows[row_idx]["reasoning"]   = match.reasoning
                output_rows[row_idx]["confidence"]  = match.confidence
                output_rows[row_idx]["needs_review"] = "" if match.confidence in ("high", "medium") else "YES"

                if plant:
                    output_rows[row_idx]["plant_id"]     = plant.id
                    output_rows[row_idx]["matched_name"]  = plant.name
                    output_rows[row_idx]["fueltype"]      = plant.fueltype
                    output_rows[row_idx]["capacity_mw"]   = plant.capacity_mw
                    print(f"  [claude]  {key.plant_name!r}  →  {plant.name!r}  (conf={match.confidence}, id={plant.id})")
                else:
                    output_rows[row_idx]["needs_review"] = "YES"
                    print(f"  [claude/unmatched]  {key.plant_name!r}  (conf={match.confidence})")

            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(output_rows[row_idx] for row_idx, _, _ in chunk)
            print(f"  Chunk {chunk_num} appended to {OUTPUT_FILE}", flush=True)

    # ── summary ───────────────────────────────────────────────────────────────
    n_claude     = sum(1 for r in output_rows if r["source"] == "claude")
    n_aggregated = sum(1 for r in output_rows if r["source"] == "aggregated")
    n_unmatched  = sum(1 for r in output_rows if r["source"] == "unmatched")
    n_review     = sum(1 for r in output_rows if r["needs_review"] == "YES")

    print(f"\nDone → {OUTPUT_FILE}")
    print(f"  Matched by Claude : {n_claude}")
    print(f"  Aggregated/skipped: {n_aggregated}")
    print(f"  Unmatched         : {n_unmatched}")
    print(f"  Needs review      : {n_review}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
