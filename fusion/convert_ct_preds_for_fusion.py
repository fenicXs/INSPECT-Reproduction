"""Convert CT classification test_preds.csv to the format expected by get_model_performance.py.

The classification pipeline outputs image_id as patient_id, but fusion expects
integer person_id and ISO datetime procedure_time. This script maps between them
using the cohort CSV.
"""

import os
import glob
import pandas as pd

# Paths
COHORT_CSV = "/scratch/pkrish52/INSPECT/data/cohort_0.2.0_master_file_anon.csv"
EXP_BASE = "/scratch/pkrish52/INSPECT/output/ct_classify_exp"
OUTPUT_DIR = "/scratch/pkrish52/INSPECT/output/ct_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build image_id -> (person_id, procedure_DATETIME) mapping from cohort
cohort = pd.read_csv(COHORT_CSV)
cohort["image_id_clean"] = cohort["image_id"].apply(
    lambda x: x.replace(".nii.gz", "") if str(x).endswith(".nii.gz") else str(x)
)
id_map = {}
for _, row in cohort.iterrows():
    id_map[row["image_id_clean"]] = (int(row["person_id"]), row["procedure_DATETIME"])

# Task name mapping (target column -> fusion filename)
task_to_filename = {
    "pe_positive_nlp": "pe_pred_proba.csv",
    "1_month_mortality": "mort_1m_pred_proba.csv",
    "6_month_mortality": "mort_6m_pred_proba.csv",
    "12_month_mortality": "mort_12m_pred_proba.csv",
    "1_month_readmission": "read_1m_pred_proba.csv",
    "6_month_readmission": "read_6m_pred_proba.csv",
    "12_month_readmission": "read_12m_pred_proba.csv",
    "12_month_PH": "ph_12m_pred_proba.csv",
}

# Find all test_preds.csv files — merge valid + test splits per task
# Group by task, collecting all pred files (from test and valid runs)
task_pred_files = {t: [] for t in task_to_filename}
pred_files = glob.glob(os.path.join(EXP_BASE, "classify_*", "test_preds.csv"))
print(f"Found {len(pred_files)} prediction files total")

for pred_file in pred_files:
    dir_name = os.path.basename(os.path.dirname(pred_file))
    for t in task_to_filename:
        if t in dir_name:
            task_pred_files[t].append(pred_file)
            break

for task, files in task_pred_files.items():
    if not files:
        print(f"  WARNING: No predictions found for {task}, skipping")
        continue

    # Merge all splits (valid + test) for this task
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Drop duplicates in case same patient appears twice
    df = df.drop_duplicates(subset=["patient_id", "procedure_time"])
    print(f"  {task}: {len(df)} predictions from {len(files)} file(s)")

    # Map image_id to person_id and procedure_DATETIME
    new_rows = []
    unmapped = 0
    for _, row in df.iterrows():
        pid_key = str(row["patient_id"])
        if pid_key in id_map:
            person_id, proc_time = id_map[pid_key]
            new_rows.append({
                "patient_id": person_id,
                "procedure_time": proc_time,
                "label": row["label"],
                "prob": row["prob"],
            })
        else:
            unmapped += 1

    if unmapped > 0:
        print(f"    WARNING: {unmapped} predictions could not be mapped")

    out_df = pd.DataFrame(new_rows)
    out_path = os.path.join(OUTPUT_DIR, task_to_filename[task])
    out_df.to_csv(out_path, index=False)
    print(f"    Saved to {out_path} ({len(out_df)} rows)")

print(f"\nAll predictions saved to {OUTPUT_DIR}")
