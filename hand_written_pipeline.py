import csv
import re
import os
import dotenv
from dataclasses import dataclass
from datetime import date
from typing import Optional, Literal
from enum import Enum

import pandas as pd

import anthropic
from pydantic import BaseModel
from rapidfuzz import process, fuzz, utils as fuzz_utils

# ── files ─────────────────────────────────────────────────────────────────────
REDISPATCH_FILE = "redispatch_top100.csv"
PLANTS_FILE     = "powerplants_pypsa_germany_merged.csv"
OUTPUT_FILE     = "plant_lookup_first100_v3.csv"
POWERPLANT_SEP = "," 

# ── tuning ────────────────────────────────────────────────────────────────────
TOP_K           = 20    # fuzzy candidates passed to LLM
AUTO_THRESHOLD  = 90    # score >= this AND unique top candidate → auto-accept
TIE_TOLERANCE   = 30    # candidates within this many points of top score count as tied
MIN_CAPACITY    = 1.0   # MW — skip plants too small for TSO redispatch (0.0 = unknown, kept)
CHUNK_SIZE      = 20    # redispatch entries per Claude API call

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

#<StringArray>
# [             'Hydro',          'Hard Coal',        'Natural Gas',
#             'Lignite',                'Oil',               'Wind',
#       'Solid Biomass',              'Waste',              'Solar',
#          'Geothermal',            'Battery',       'Heat Storage',
#             'Nuclear',              'Other',             'Biogas',
#  'Mechanical Storage',   'Hydrogen Storage']
class powerPlantFuelType(Enum):
    NATURAL_GAS = "Natural Gas"
    HARD_COAL = "Hard Coal"
    LIGNITE = "Lignite"
    NUCLEAR = "Nuclear"
    OIL = "Oil"
    WASTE = "Waste"
    BIOGAS = "Biogas"
    SOLID_BIOMASS = "Solid Biomass"
    GEOTHERMAL = "Geothermal"
    OTHER = "Other"
    HYDRO = "Hydro"
    SOLAR = "Solar"
    WIND = "Wind"
    MECHANICAL_STORAGE = "Mechanical Storage"
    HYDROGEN_STORAGE = "Hydrogen Storage"
    HEAT_STORAGE = "Heat Storage"
    BATTERY = "Battery"
    HARD_COAL = "Hard Coal"

@dataclass(frozen=True) # will be used as dict keys
class RedispatchKey:
    plant_name: str
    energy_type: str
    instructing_tso: str
    begin_date: date

@dataclass
class PowerPlant:
    id: int
    name: str
    fueltype: powerPlantFuelType
    technology: str
    