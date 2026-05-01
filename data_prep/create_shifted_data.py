"""
Create time-shifted timeline CSVs and labeled_patients for MOTOR batch creation.
Shifts all dates by +100 years to avoid pre-1970 epoch bug in femr C++ extension.

The MOTOR model uses relative time encodings, so the shift doesn't affect representations.
"""

import csv
import os
import sys
import glob
import datetime

csv.field_size_limit(sys.maxsize)

SHIFT_YEARS = 100
SHIFT = datetime.timedelta(days=SHIFT_YEARS * 365)  # Approximate

TIMELINE_DIR = "/scratch/pkrish52/INSPECT/data/timelines_smallfiles"
SHIFTED_DIR = "/scratch/pkrish52/INSPECT/data/timelines_shifted"
LABELS_DIR = "/scratch/pkrish52/INSPECT/output/labels_and_features"
SHIFTED_LABELS_DIR = "/scratch/pkrish52/INSPECT/output/labels_and_features_shifted"

os.makedirs(SHIFTED_DIR, exist_ok=True)
os.makedirs(SHIFTED_LABELS_DIR, exist_ok=True)


def shift_datetime(dt_str):
    """Shift a datetime string by +100 years."""
    if not dt_str or dt_str == "":
        return dt_str
    try:
        dt = datetime.datetime.fromisoformat(dt_str)
        shifted = dt + SHIFT
        return shifted.isoformat(sep="T")
    except (ValueError, TypeError):
        return dt_str


# Step 1: Create shifted timeline CSVs
print("=== Step 1: Shifting timeline CSVs ===")
for fpath in sorted(glob.glob(os.path.join(TIMELINE_DIR, "timelines_*.csv"))):
    fname = os.path.basename(fpath)
    outpath = os.path.join(SHIFTED_DIR, fname)
    print(f"  {fname}...", end=" ", flush=True)

    with open(fpath) as fin, open(outpath, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            row["start"] = shift_datetime(row["start"])
            writer.writerow(row)
            count += 1
    print(f"{count} rows")

# Step 2: Create shifted labeled_patients CSVs
print("\n=== Step 2: Shifting labeled_patients CSVs ===")
tasks = [
    "12_month_mortality", "6_month_mortality", "1_month_mortality",
    "1_month_readmission", "6_month_readmission", "12_month_readmission",
    "12_month_PH", "PE",
]

for task in tasks:
    src = os.path.join(LABELS_DIR, task, "labeled_patients.csv")
    dst_dir = os.path.join(SHIFTED_LABELS_DIR, task)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "labeled_patients.csv")

    if not os.path.exists(src):
        print(f"  {task}: MISSING")
        continue

    with open(src) as fin, open(dst, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            row["prediction_time"] = shift_datetime(row["prediction_time"])
            writer.writerow(row)
            count += 1
    print(f"  {task}: {count} labels shifted")

print("\nDone!")
print(f"Shifted timelines in: {SHIFTED_DIR}")
print(f"Shifted labels in: {SHIFTED_LABELS_DIR}")
