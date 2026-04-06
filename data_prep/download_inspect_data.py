"""
Download INSPECT EHR data from Redivis and convert OMOP tables
to flat timeline CSVs for etl_simple_femr.

etl_simple_femr expects CSVs with columns: patient_id, time, code [, numeric_value]

Usage:
    export REDIVIS_API_TOKEN="your_token_here"
    python download_inspect_data.py

Output:
    data/timelines_smallfiles/   - Timeline CSVs for etl_simple_femr
"""

import os
import gc
import redivis
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/scratch/pkrish52/INSPECT/data")
TIMELINES_DIR = BASE_DIR / "timelines_smallfiles"
TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

dataset = redivis.user("shahlab").dataset("inspect_ehr")


def download_table(table_name, max_results=None):
    """Download a Redivis table as a pandas DataFrame."""
    print(f"  Downloading: {table_name}...", end=" ", flush=True)
    table = dataset.table(table_name)
    df = table.to_pandas_dataframe(max_results=max_results)
    print(f"{len(df)} rows")
    return df


# ──────────────────────────────────────────────────────────────
# Step 1: Build concept_id -> code mapping
# ──────────────────────────────────────────────────────────────
print("=== Step 1: Build concept mapping ===")
concept_df = download_table("concept")
concept_map = {}
for _, row in concept_df.iterrows():
    cid = row.get("concept_id")
    vocab = row.get("vocabulary_id", "")
    code = row.get("concept_code", "")
    if pd.notna(cid) and pd.notna(vocab) and pd.notna(code):
        concept_map[int(cid)] = f"{vocab}/{code}"
print(f"  Mapped {len(concept_map)} concepts")
del concept_df
gc.collect()

# ──────────────────────────────────────────────────────────────
# Step 2: Process each OMOP table into timeline format
# ──────────────────────────────────────────────────────────────

# Table configs: (table_name, person_col, time_col, concept_col, value_col_or_None)
TABLE_CONFIGS = [
    ("condition_occurrence", "person_id", "condition_start_DATETIME", "condition_concept_id", None),
    ("drug_exposure", "person_id", "drug_exposure_start_DATETIME", "drug_concept_id", None),
    ("procedure_occurrence", "person_id", "procedure_DATETIME", "procedure_concept_id", None),
    ("observation", "person_id", "observation_DATETIME", "observation_concept_id", "value_as_number"),
    ("visit_occurrence", "person_id", "visit_start_DATETIME", "visit_concept_id", None),
    ("death", "person_id", "death_DATETIME", "death_type_concept_id", None),
    ("device_exposure", "person_id", "device_exposure_start_DATETIME", "device_concept_id", None),
]

all_timeline_dfs = []
file_counter = 0


def convert_table(table_name, person_col, time_col, concept_col, value_col):
    """Convert an OMOP table to timeline format using vectorized operations."""
    print(f"\n=== Processing: {table_name} ===")
    df = download_table(table_name)

    # Handle column name case differences
    col_lower = {c.lower(): c for c in df.columns}

    def find(target):
        if target in df.columns:
            return target
        t = target.lower()
        if t in col_lower:
            return col_lower[t]
        # Try date variant
        alt = target.replace("_DATETIME", "_DATE").replace("_datetime", "_date")
        if alt in df.columns:
            return alt
        if alt.lower() in col_lower:
            return col_lower[alt.lower()]
        return None

    pcol = find(person_col)
    tcol = find(time_col)
    ccol = find(concept_col)
    vcol = find(value_col) if value_col else None

    if pcol is None or ccol is None:
        print(f"  SKIP: Missing columns. Available: {list(df.columns)}")
        return pd.DataFrame()

    # Filter valid rows
    mask = df[ccol].notna() & (df[ccol] != 0)
    if pcol:
        mask &= df[pcol].notna()
    df = df[mask].copy()

    # Map concept IDs to codes
    df["code"] = df[ccol].astype(int).map(concept_map)
    df = df[df["code"].notna()]

    # Build result
    result = pd.DataFrame({
        "patient_id": df[pcol].astype(int),
        "time": df[tcol] if tcol else "",
        "code": df["code"],
    })

    if vcol and vcol in df.columns:
        result["numeric_value"] = df[vcol]

    print(f"  -> {len(result)} timeline entries")
    return result


# Process each table
for config in TABLE_CONFIGS:
    tl = convert_table(*config)
    if len(tl) > 0:
        all_timeline_dfs.append(tl)
    gc.collect()

# ──────────────────────────────────────────────────────────────
# Step 3: Measurement table (161M rows - process in chunks)
# ──────────────────────────────────────────────────────────────
print("\n=== Processing: measurement (large table - chunked) ===")

# Get unique patient IDs from our cohort
cohort = pd.read_csv(BASE_DIR / "cohort_0.2.0_master_file_anon.csv")
cohort_pids = set(cohort["PatientID"].unique())
print(f"  Cohort has {len(cohort_pids)} unique patients")

# Download measurement table in patient batches to manage memory
patient_list = sorted(list(cohort_pids))
BATCH_SIZE = 2000
measurement_dfs = []

for i in range(0, len(patient_list), BATCH_SIZE):
    batch_pids = patient_list[i : i + BATCH_SIZE]
    pids_str = ",".join(str(p) for p in batch_pids)
    print(f"  Batch {i // BATCH_SIZE + 1}/{(len(patient_list) + BATCH_SIZE - 1) // BATCH_SIZE}: patients {i}-{i + len(batch_pids)}...", end=" ", flush=True)

    query = dataset.query(f"""
        SELECT person_id, measurement_DATETIME, measurement_concept_id, value_as_number
        FROM measurement
        WHERE person_id IN ({pids_str})
          AND measurement_concept_id IS NOT NULL
          AND measurement_concept_id != 0
    """)
    mdf = query.to_pandas_dataframe()
    print(f"{len(mdf)} rows")

    if len(mdf) > 0:
        mdf["code"] = mdf["measurement_concept_id"].astype(int).map(concept_map)
        mdf = mdf[mdf["code"].notna()]
        result = pd.DataFrame({
            "patient_id": mdf["person_id"].astype(int),
            "time": mdf["measurement_DATETIME"],
            "code": mdf["code"],
            "numeric_value": mdf["value_as_number"],
        })
        measurement_dfs.append(result)
    gc.collect()

if measurement_dfs:
    measurement_combined = pd.concat(measurement_dfs, ignore_index=True)
    print(f"  Total measurement entries: {len(measurement_combined)}")
    all_timeline_dfs.append(measurement_combined)
    del measurement_dfs, measurement_combined
    gc.collect()

# ──────────────────────────────────────────────────────────────
# Step 4: Add demographics from person table
# ──────────────────────────────────────────────────────────────
print("\n=== Processing: demographics ===")
person_df = download_table("person")

demo_rows = []
for _, row in person_df.iterrows():
    pid = row.get("person_id")
    if pd.isna(pid):
        continue
    pid = int(pid)

    # Birth
    year = row.get("year_of_birth")
    month = row.get("month_of_birth", 1)
    day = row.get("day_of_birth", 1)
    if pd.notna(year):
        import datetime
        birth_time = datetime.datetime(
            int(year),
            int(month) if pd.notna(month) else 1,
            int(day) if pd.notna(day) else 1,
        )
        demo_rows.append({"patient_id": pid, "time": birth_time, "code": "Birth"})

    # Gender
    gid = row.get("gender_concept_id")
    if pd.notna(gid) and int(gid) != 0:
        demo_rows.append({"patient_id": pid, "time": "", "code": concept_map.get(int(gid), f"OMOP/{int(gid)}")})

    # Race
    rid = row.get("race_concept_id")
    if pd.notna(rid) and int(rid) != 0:
        demo_rows.append({"patient_id": pid, "time": "", "code": concept_map.get(int(rid), f"OMOP/{int(rid)}")})

    # Ethnicity
    eid = row.get("ethnicity_concept_id")
    if pd.notna(eid) and int(eid) != 0:
        demo_rows.append({"patient_id": pid, "time": "", "code": concept_map.get(int(eid), f"OMOP/{int(eid)}")})

demo_df = pd.DataFrame(demo_rows)
print(f"  {len(demo_df)} demographic entries")
all_timeline_dfs.append(demo_df)
del person_df, demo_rows
gc.collect()

# ──────────────────────────────────────────────────────────────
# Step 5: Combine and write timeline files
# ──────────────────────────────────────────────────────────────
print("\n=== Combining all timelines ===")
combined = pd.concat(all_timeline_dfs, ignore_index=True)
del all_timeline_dfs
gc.collect()
print(f"Total entries: {len(combined)}")

# Ensure consistent columns
if "numeric_value" not in combined.columns:
    combined["numeric_value"] = np.nan

# Sort by patient_id and time
combined = combined.sort_values(["patient_id", "time"]).reset_index(drop=True)

# Write in chunks of ~1000 patients
print("\n=== Writing timeline files ===")
patient_ids = sorted(combined["patient_id"].unique())
CHUNK_SIZE = 1000

for i in range(0, len(patient_ids), CHUNK_SIZE):
    chunk_pids = patient_ids[i : i + CHUNK_SIZE]
    chunk = combined[combined["patient_id"].isin(set(chunk_pids))]
    fname = TIMELINES_DIR / f"timelines_{i // CHUNK_SIZE:04d}.csv"
    chunk.to_csv(fname, index=False)
    print(f"  {fname.name}: {len(chunk)} entries, {len(chunk_pids)} patients")

total_files = len(list(TIMELINES_DIR.glob("*.csv")))
print(f"\n=== Done! {total_files} timeline files in {TIMELINES_DIR} ===")
