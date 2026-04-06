"""Prune Athena ontology files to only include concepts used in our timeline data.
This dramatically reduces memory usage during etl_simple_femr."""

import csv
import os
import sys
import glob

csv.field_size_limit(sys.maxsize)

TIMELINE_DIR = "/scratch/pkrish52/INSPECT/data/timelines_smallfiles"
ATHENA_DIR = "/scratch/pkrish52/INSPECT/ATHENA"
PRUNED_DIR = "/scratch/pkrish52/INSPECT/ATHENA_pruned"

os.makedirs(PRUNED_DIR, exist_ok=True)

# Step 1: Collect all unique vocabulary_id/concept_code pairs from timeline data
print("Step 1: Collecting codes from timeline data...")
codes_in_data = set()
for fpath in sorted(glob.glob(os.path.join(TIMELINE_DIR, "timelines_*.csv"))):
    print(f"  Reading {os.path.basename(fpath)}...")
    with open(fpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            codes_in_data.add(row["code"])
print(f"  Found {len(codes_in_data)} unique codes in timeline data")

# Step 2: Find matching concept_ids from CONCEPT.csv
print("Step 2: Mapping codes to concept_ids from CONCEPT.csv...")
concept_ids_used = set()
concept_rows_kept = 0
concept_rows_total = 0

with open(os.path.join(ATHENA_DIR, "CONCEPT.csv")) as fin, \
     open(os.path.join(PRUNED_DIR, "CONCEPT.csv"), "w") as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    for row in reader:
        concept_rows_total += 1
        key = f'{row["vocabulary_id"]}/{row["concept_code"]}'
        if key in codes_in_data:
            writer.writerow(row)
            concept_ids_used.add(row["concept_id"])
            concept_rows_kept += 1
        if concept_rows_total % 1_000_000 == 0:
            print(f"  Processed {concept_rows_total} concepts, kept {concept_rows_kept}...")

print(f"  CONCEPT.csv: {concept_rows_total} -> {concept_rows_kept} rows ({len(concept_ids_used)} concept_ids)")

# Step 3: Prune CONCEPT_RELATIONSHIP.csv to only include used concept_ids
print("Step 3: Pruning CONCEPT_RELATIONSHIP.csv...")
rel_kept = 0
rel_total = 0

with open(os.path.join(ATHENA_DIR, "CONCEPT_RELATIONSHIP.csv")) as fin, \
     open(os.path.join(PRUNED_DIR, "CONCEPT_RELATIONSHIP.csv"), "w") as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    for row in reader:
        rel_total += 1
        if row["concept_id_1"] in concept_ids_used or row["concept_id_2"] in concept_ids_used:
            writer.writerow(row)
            rel_kept += 1
        if rel_total % 5_000_000 == 0:
            print(f"  Processed {rel_total} relationships, kept {rel_kept}...")

print(f"  CONCEPT_RELATIONSHIP.csv: {rel_total} -> {rel_kept} rows")

# Step 4: Copy other small files as-is
for fname in ["CONCEPT_CLASS.csv", "DOMAIN.csv", "RELATIONSHIP.csv", "VOCABULARY.csv",
              "CONCEPT_ANCESTOR.csv", "CONCEPT_SYNONYM.csv", "DRUG_STRENGTH.csv", "CONCEPT_CPT4.csv"]:
    src = os.path.join(ATHENA_DIR, fname)
    dst = os.path.join(PRUNED_DIR, fname)
    if os.path.exists(src):
        # For large files, just create empty version; for small files, copy
        size_mb = os.path.getsize(src) / (1024*1024)
        if size_mb > 100:
            print(f"  Skipping {fname} ({size_mb:.0f}MB) - not needed by etl_simple_femr")
        else:
            print(f"  Copying {fname} ({size_mb:.1f}MB)")
            with open(src) as fin, open(dst, "w") as fout:
                fout.write(fin.read())

print("\nDone! Pruned Athena files written to:", PRUNED_DIR)
print(f"Original CONCEPT_RELATIONSHIP.csv: {os.path.getsize(os.path.join(ATHENA_DIR, 'CONCEPT_RELATIONSHIP.csv')) / (1024*1024):.0f}MB")
print(f"Pruned CONCEPT_RELATIONSHIP.csv: {os.path.getsize(os.path.join(PRUNED_DIR, 'CONCEPT_RELATIONSHIP.csv')) / (1024*1024):.0f}MB")
