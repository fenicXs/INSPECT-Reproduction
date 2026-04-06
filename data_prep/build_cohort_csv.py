"""
Build cohort_0.2.0_master_file_anon.csv from Redivis OMOP tables.

This generates the cohort CSV needed by the INSPECT_public EHR pipeline:
  - PatientID: person_id from crosswalk
  - StudyTime: note_DATETIME from the note table (CT study time)
  - split: train/valid/test (hash-based, ~70/10/20 split)
  - pe_positive_nlp: PE diagnosis within ±1 day of study (proxy for NLP label)
  - 12_month_PH: PH diagnosis within 12 months after study (True/False/Censored)

Usage:
    export REDIVIS_API_TOKEN="your_token_here"
    python build_cohort_csv.py
"""

import os
import hashlib
import datetime
import redivis
import pandas as pd
import numpy as np

DATA_DIR = "/scratch/pkrish52/INSPECT/data"
os.makedirs(DATA_DIR, exist_ok=True)

dataset = redivis.user("shahlab").dataset("inspect_ehr")

# ──────────────────────────────────────────────────────────────
# Step 1: Load crosswalk and note times
# ──────────────────────────────────────────────────────────────
print("=== Step 1: Load crosswalk + note times ===")

crosswalk = pd.read_csv(os.path.join(DATA_DIR, "image_ehr_crosswalk.csv"))
print(f"Crosswalk: {len(crosswalk)} studies, {crosswalk['person_id'].nunique()} patients")

# Load previously downloaded cohort notes
notes_path = os.path.join(DATA_DIR, "cohort_notes.csv")
if os.path.exists(notes_path):
    notes = pd.read_csv(notes_path)
else:
    note_ids = crosswalk["note_id"].tolist()
    query = dataset.query(
        "SELECT note_id, person_id, note_DATETIME FROM note WHERE note_id IN ("
        + ",".join(str(n) for n in note_ids)
        + ")"
    )
    notes = query.to_pandas_dataframe()
    notes.to_csv(notes_path, index=False)

print(f"Notes: {len(notes)} records")

# Merge crosswalk with note times
cohort = crosswalk.merge(
    notes[["note_id", "note_DATETIME"]],
    on="note_id",
    how="left",
)
cohort = cohort.rename(
    columns={"person_id": "PatientID", "note_DATETIME": "StudyTime"}
)
cohort = cohort.dropna(subset=["StudyTime"])
print(f"After merge: {len(cohort)} studies with valid StudyTime")

# ──────────────────────────────────────────────────────────────
# Step 2: Generate train/valid/test splits (per patient)
# ──────────────────────────────────────────────────────────────
print("\n=== Step 2: Generate splits ===")


def hash_split(patient_id, seed=42):
    """Deterministic hash-based split: ~70% train, 10% valid, 20% test."""
    h = hashlib.md5(f"{patient_id}_{seed}".encode()).hexdigest()
    val = int(h[:8], 16) / (16**8)
    if val < 0.7:
        return "train"
    elif val < 0.8:
        return "valid"
    else:
        return "test"


patient_ids = cohort["PatientID"].unique()
split_map = {pid: hash_split(pid) for pid in patient_ids}
cohort["split"] = cohort["PatientID"].map(split_map)

split_counts = cohort["split"].value_counts()
print(f"Split distribution:\n{split_counts}")

# ──────────────────────────────────────────────────────────────
# Step 3: Derive PE label from condition_occurrence
# ──────────────────────────────────────────────────────────────
print("\n=== Step 3: Derive PE labels ===")

# PE-related concept IDs (ICD10CM I26.*, ICD9CM 415.1*, SNOMED 59282003)
pe_concept_ids = [
    440417,  # SNOMED: Pulmonary embolism (59282003)
    45605789,  # ICD10CM I26.90
    45586576,  # ICD10CM I26.09
    45572083,  # ICD10CM I26.01
    44819706,  # ICD9CM 415.19
    45552786,  # ICD10CM I26.92
    44819705,  # ICD9CM 415.12
    45557540,  # ICD10CM I26.02
    45572084,  # ICD10CM I26.99
    44823112,  # ICD9CM 415.11
    44825436,  # ICD9CM 415.13
    1553749,  # ICD10CM I26.93
    1569147,  # ICD10CM I26
    44819704,  # ICD9CM 415.1
    1553750,  # ICD10CM I26.94
    1569149,  # ICD10CM I26.9
    1569148,  # ICD10CM I26.0
]

# Also get descendants of PE concept via concept_ancestor
print("  Querying PE descendants...")
pe_query = dataset.query(f"""
    SELECT DISTINCT descendant_concept_id
    FROM concept_ancestor
    WHERE ancestor_concept_id = 440417
""")
pe_descendants = pe_query.to_pandas_dataframe()
all_pe_ids = set(pe_concept_ids) | set(pe_descendants["descendant_concept_id"].tolist())
print(f"  Total PE concept IDs (including descendants): {len(all_pe_ids)}")

# Get PE diagnoses for our patients
patient_list = ",".join(str(p) for p in patient_ids)
pe_ids_str = ",".join(str(c) for c in all_pe_ids)

print("  Querying PE conditions...")
pe_cond_query = dataset.query(f"""
    SELECT person_id, condition_start_DATETIME
    FROM condition_occurrence
    WHERE person_id IN ({patient_list})
      AND condition_concept_id IN ({pe_ids_str})
""")
pe_conds = pe_cond_query.to_pandas_dataframe()
print(f"  Found {len(pe_conds)} PE condition records for {pe_conds['person_id'].nunique()} patients")

# For each study, check if PE diagnosis within ±1 day of StudyTime
pe_conds["condition_start_DATETIME"] = pd.to_datetime(pe_conds["condition_start_DATETIME"])
cohort["StudyTime_dt"] = pd.to_datetime(cohort["StudyTime"])

pe_labels = []
for _, row in cohort.iterrows():
    pid = row["PatientID"]
    study_time = row["StudyTime_dt"]
    patient_pe = pe_conds[pe_conds["person_id"] == pid]
    if len(patient_pe) > 0:
        # Check if any PE diagnosis within ±1 day of study
        time_diffs = (patient_pe["condition_start_DATETIME"] - study_time).abs()
        has_pe = (time_diffs <= pd.Timedelta(days=1)).any()
    else:
        has_pe = False
    pe_labels.append(has_pe)

cohort["pe_positive_nlp"] = pe_labels
pe_pos = sum(pe_labels)
print(f"  PE positive studies: {pe_pos} / {len(cohort)} ({100*pe_pos/len(cohort):.1f}%)")

# ──────────────────────────────────────────────────────────────
# Step 4: Derive 12-month PH label from condition_occurrence
# ──────────────────────────────────────────────────────────────
print("\n=== Step 4: Derive PH labels ===")

# PH codes from INSPECT_public/ehr/2_generate_labels_and_features.py
PH_icd_codes = [
    "I27.21", "I27.22", "416", "416.1", "416.2", "I27.2", "I27.29",
    "I27.83", "I27.0", "416.0", "416.8", "I27.89", "I27.82", "I27.1",
    "I27.20", "I27.23", "416.9", "I27.81",
]

# Get PH concept IDs
ph_codes_str = ",".join(f"'{c}'" for c in PH_icd_codes)
ph_concept_query = dataset.query(f"""
    SELECT DISTINCT concept_id
    FROM concept
    WHERE concept_code IN ({ph_codes_str})
      AND vocabulary_id IN ('ICD10CM', 'ICD9CM')
""")
ph_concepts = ph_concept_query.to_pandas_dataframe()
ph_concept_ids = set(ph_concepts["concept_id"].tolist())

# Also get SNOMED PH concept 70995007 and descendants
ph_desc_query = dataset.query("""
    SELECT DISTINCT descendant_concept_id
    FROM concept_ancestor
    WHERE ancestor_concept_id IN (
        SELECT concept_id FROM concept
        WHERE concept_code = '70995007' AND vocabulary_id = 'SNOMED'
    )
""")
ph_descendants = ph_desc_query.to_pandas_dataframe()
all_ph_ids = ph_concept_ids | set(ph_descendants["descendant_concept_id"].tolist())
print(f"  Total PH concept IDs: {len(all_ph_ids)}")

ph_ids_str = ",".join(str(c) for c in all_ph_ids)

print("  Querying PH conditions...")
ph_cond_query = dataset.query(f"""
    SELECT person_id, condition_start_DATETIME
    FROM condition_occurrence
    WHERE person_id IN ({patient_list})
      AND condition_concept_id IN ({ph_ids_str})
""")
ph_conds = ph_cond_query.to_pandas_dataframe()
print(f"  Found {len(ph_conds)} PH condition records for {ph_conds['person_id'].nunique()} patients")

ph_conds["condition_start_DATETIME"] = pd.to_datetime(ph_conds["condition_start_DATETIME"])

# Check if PH diagnosis within 12 months after study
# Also need to check if patient has enough follow-up (not censored)
# For simplicity: True if PH within 365 days, False if no PH and >= 365 days follow-up, Censored otherwise

# Get last known event per patient (from death or last observation)
print("  Querying death dates...")
death_query = dataset.query(f"""
    SELECT person_id, death_DATETIME
    FROM death
    WHERE person_id IN ({patient_list})
""")
death_df = death_query.to_pandas_dataframe()
death_map = {}
for _, row in death_df.iterrows():
    death_map[row["person_id"]] = pd.to_datetime(row["death_DATETIME"])

# Use observation_period for follow-up endpoint
print("  Querying observation periods...")
obs_query = dataset.query(f"""
    SELECT person_id, MAX(observation_period_end_DATE) as last_obs
    FROM observation_period
    WHERE person_id IN ({patient_list})
    GROUP BY person_id
""")
obs_df = obs_query.to_pandas_dataframe()
obs_map = {}
for _, row in obs_df.iterrows():
    obs_map[row["person_id"]] = pd.to_datetime(row["last_obs"])

ph_labels = []
for _, row in cohort.iterrows():
    pid = row["PatientID"]
    study_time = row["StudyTime_dt"]
    horizon_end = study_time + pd.Timedelta(days=365)

    patient_ph = ph_conds[ph_conds["person_id"] == pid]

    # Check for PH within 12 months
    if len(patient_ph) > 0:
        future_ph = patient_ph[
            (patient_ph["condition_start_DATETIME"] > study_time)
            & (patient_ph["condition_start_DATETIME"] <= horizon_end)
        ]
        if len(future_ph) > 0:
            ph_labels.append("True")
            continue

    # Check censoring: do we have enough follow-up?
    last_known = death_map.get(pid) or obs_map.get(pid)
    if last_known is not None and last_known >= horizon_end:
        ph_labels.append("False")
    else:
        ph_labels.append("Censored")

cohort["12_month_PH"] = ph_labels
ph_dist = pd.Series(ph_labels).value_counts()
print(f"  PH distribution:\n{ph_dist}")

# ──────────────────────────────────────────────────────────────
# Step 5: Save cohort CSV
# ──────────────────────────────────────────────────────────────
print("\n=== Step 5: Save cohort CSV ===")

output_cols = ["PatientID", "StudyTime", "split", "pe_positive_nlp", "12_month_PH"]
output = cohort[output_cols].copy()
output_path = os.path.join(DATA_DIR, "cohort_0.2.0_master_file_anon.csv")
output.to_csv(output_path, index=False)
print(f"Saved {len(output)} rows to {output_path}")
print(f"Columns: {list(output.columns)}")
print(f"\nSample:")
print(output.head())
print(f"\nSplit distribution:")
print(output["split"].value_counts())
print(f"\nPE positive: {(output['pe_positive_nlp'] == True).sum()}")
print(f"PH True/False/Censored: {output['12_month_PH'].value_counts().to_dict()}")
