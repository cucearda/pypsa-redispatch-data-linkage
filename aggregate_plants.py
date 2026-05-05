"""
aggregate_plants.py

Preprocessing step: merge individual turbine entries that share the same name
and are geographically co-located into a single farm entry.

Output row per merged group:
  - id           : representative id (first by original row order)
  - Capacity     : summed across all turbines
  - lat / lon    : centroid of turbines that have coordinates
  - turbine_count: number of turbines merged (1 = not merged)
  - source_ids   : comma-separated original ids

Groups whose members span more than MAX_FARM_RADIUS_KM are NOT merged
(different plants that happen to share a name).

Output: powerplants_pypsa_germany_merged.csv
"""

import math
import os
import pandas as pd

INPUT_FILE         = "powerplants_pypsa_germany.csv"
OUTPUT_FILE        = "powerplants_pypsa_germany_merged.csv"
MAX_FARM_RADIUS_KM = 50.0   # max distance from centroid to any turbine to allow merge


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def merge_group(group: pd.DataFrame) -> dict:
    rep = group.iloc[0].to_dict()
    rep["Capacity"] = group["Capacity"].sum()

    with_coords = group.dropna(subset=["lat", "lon"])
    if len(with_coords) > 0:
        rep["lat"] = with_coords["lat"].mean()
        rep["lon"] = with_coords["lon"].mean()

    rep["turbine_count"] = len(group)
    rep["source_ids"]    = ",".join(str(i) for i in group["id"].tolist())
    return rep


def max_dist_from_centroid(group: pd.DataFrame) -> float:
    with_coords = group.dropna(subset=["lat", "lon"])
    if len(with_coords) < 2:
        return 0.0
    clat = with_coords["lat"].mean()
    clon = with_coords["lon"].mean()
    return max(
        haversine_km(r["lat"], r["lon"], clat, clon)
        for _, r in with_coords.iterrows()
    )


def main() -> None:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    df["lat"]      = pd.to_numeric(df["lat"],      errors="coerce")
    df["lon"]      = pd.to_numeric(df["lon"],      errors="coerce")
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce").fillna(0.0)
    df["Name"]     = df["Name"].str.strip()

    df["turbine_count"] = 1
    df["source_ids"]    = df["id"].astype(str)

    merged_rows      = []
    n_merged_groups  = 0
    n_merged_turbines = 0
    n_skipped_groups = 0

    for name, group in df.groupby("Name", sort=False):
        if len(group) == 1:
            merged_rows.append(group.iloc[0].to_dict())
            continue

        dist = max_dist_from_centroid(group)

        if dist <= MAX_FARM_RADIUS_KM:
            merged_rows.append(merge_group(group))
            n_merged_groups  += 1
            n_merged_turbines += len(group)
            print(f"  merged  {len(group):3d} × '{name}'  "
                  f"(max_dist={dist:.1f} km, total={group['Capacity'].sum():.1f} MW)")
        else:
            for _, r in group.iterrows():
                merged_rows.append(r.to_dict())
            n_skipped_groups += 1
            print(f"  skipped {len(group):3d} × '{name}'  "
                  f"(span={dist:.1f} km > {MAX_FARM_RADIUS_KM} km — kept separate)")

    out = pd.DataFrame(merged_rows).reset_index(drop=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nDone → {OUTPUT_FILE}")
    print(f"  Input rows     : {len(df)}")
    print(f"  Output rows    : {len(out)}  (reduced by {len(df) - len(out)})")
    print(f"  Merged groups  : {n_merged_groups}  ({n_merged_turbines} turbines → {n_merged_groups} entries)")
    print(f"  Skipped (spread too wide): {n_skipped_groups}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
