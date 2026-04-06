"""Generate metadata CSVs and slice thickness dictionary for RadFusion3 pipeline."""

import pandas as pd
import numpy as np
import nibabel as nib
import pickle
import os
from pathlib import Path
from tqdm import tqdm

# Paths
COHORT_CSV = "/scratch/pkrish52/INSPECT/data/cohort_0.2.0_master_file_anon.csv"
CTPA_DIR = "/scratch/pkrish52/INSPECT/data/CTPA/inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/full/CTPA"
OUTPUT_DIR = "/scratch/pkrish52/INSPECT/data/image_pipeline"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read cohort
df = pd.read_csv(COHORT_CSV)
print(f"Cohort: {len(df)} rows, {df['impression_id'].nunique()} unique impressions")

# Filter to only rows with CTPA images available
available_images = set(f.name for f in Path(CTPA_DIR).glob("*.nii.gz"))
df["has_image"] = df["image_id"].apply(lambda x: f"{x}.nii.gz" in available_images if not str(x).endswith(".nii.gz") else x in available_images)

# Fix image_id to include .nii.gz if not already
df["image_id"] = df["image_id"].apply(lambda x: x if str(x).endswith(".nii.gz") else f"{x}.nii.gz")
df["has_image"] = df["image_id"].apply(lambda x: x in available_images)

print(f"Rows with available CTPA images: {df['has_image'].sum()} / {len(df)}")

df_img = df[df["has_image"]].copy()

# 1. Final_metadata.csv
metadata_cols = ["impression_id", "image_id", "person_id", "procedure_DATETIME"]
df_metadata = df_img[metadata_cols].copy()
df_metadata.to_csv(os.path.join(OUTPUT_DIR, "Final_metadata.csv"), index=False)
print(f"Final_metadata.csv: {len(df_metadata)} rows")

# 2. Final_labels.csv
label_cols = ["impression_id", "pe_positive_nlp", "1_month_mortality", "6_month_mortality",
              "12_month_mortality", "1_month_readmission", "6_month_readmission",
              "12_month_readmission"]
# Check if 12_month_PH exists
if "12_month_PH" in df_img.columns:
    label_cols.append("12_month_PH")
elif "12_month_ph" in df_img.columns:
    label_cols.append("12_month_ph")

df_labels = df_img[label_cols].copy()
df_labels.to_csv(os.path.join(OUTPUT_DIR, "Final_labels.csv"), index=False)
print(f"Final_labels.csv: {len(df_labels)} rows")

# 3. Final_splits.csv
df_splits = df_img[["impression_id", "split"]].copy()
df_splits.to_csv(os.path.join(OUTPUT_DIR, "Final_splits.csv"), index=False)
print(f"Final_splits.csv: {len(df_splits)} rows")
print(f"Split distribution:\n{df_splits['split'].value_counts()}")

# 4. Generate slice thickness dictionary from NIfTI headers
print("\nGenerating slice thickness dictionary from NIfTI headers...")
dict_slice_thickness = {}
errors = 0

for _, row in tqdm(df_img.iterrows(), total=len(df_img)):
    image_id = row["image_id"]
    key = image_id.replace(".nii.gz", "")
    nifti_path = os.path.join(CTPA_DIR, image_id)
    try:
        img = nib.load(nifti_path)
        zooms = img.header.get_zooms()
        # Slice thickness is the 3rd dimension
        if len(zooms) >= 3:
            dict_slice_thickness[key] = float(zooms[2])
        else:
            dict_slice_thickness[key] = 1.0  # default
    except Exception as e:
        errors += 1
        dict_slice_thickness[key] = 1.0  # default
        if errors <= 5:
            print(f"  Error reading {image_id}: {e}")

with open(os.path.join(OUTPUT_DIR, "dict_slice_thickness.pkl"), "wb") as f:
    pickle.dump(dict_slice_thickness, f)

print(f"dict_slice_thickness.pkl: {len(dict_slice_thickness)} entries, {errors} errors")
print(f"\nAll files saved to {OUTPUT_DIR}")
