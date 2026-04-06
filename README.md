# INSPECT Benchmark Reproduction

Reproduction of the INSPECT multimodal benchmark (arXiv:2311.10798) for pulmonary embolism diagnosis and prognosis.

**Tasks:** PE detection, 1/6/12-month mortality, 1/6/12-month readmission, 12-month pulmonary hypertension  
**Modalities:** EHR (MOTOR + GBM) + CT Image (ResNetV2 + GRU/LSTM) + Late Fusion

---

## Repository Structure

```
├── data_prep/          # Data download and preprocessing
├── ehr/                # EHR pipeline (GBM + MOTOR)
├── image/              # CT image pipeline (featurization + classification)
│   └── radfusion3/     # Model configs, datasets, lightning modules
├── fusion/             # Late fusion and result evaluation
└── scripts/            # End-to-end run scripts
```

---

## Requirements

### Data
- EHR cohort CSV: downloaded via Redivis API (`data_prep/download_inspect_data.py`)
- CT scans (~2TB NIfTI): downloaded from Stanford Azure storage
- Athena OMOP vocabularies: downloaded from OHDSI Athena
- Pre-trained MOTOR-T-Base checkpoint: HuggingFace `StanfordShahLab/motor-t-base`
- Pre-trained ResNetV2-CT checkpoint: HuggingFace `StanfordShahLab/inspect`

### Environments
- **EHR:** `inspect_ehr` conda env (FEMR, LightGBM, PyTorch)
- **Image:** `radfusion3` conda env (PyTorch 1.11, PyTorch Lightning 2.0, timm 0.9.1)

---

## Reproduction Steps

### Step 1 — Data Preparation
```bash
# Download EHR data via Redivis
python data_prep/download_inspect_data.py

# Prune Athena ontology (reduces ~15GB to manageable size)
python data_prep/prune_athena.py \
    --athena_dir /path/to/ATHENA \
    --output_dir /path/to/ATHENA_pruned \
    --cohort_csv /path/to/cohort_0.2.0_master_file_anon.csv

# Generate FEMR patient database
python data_prep/run_etl.py

# Generate CT image metadata files and slice thickness dictionary
python data_prep/prep_image_pipeline.py
```

### Step 2 — EHR Pipeline

#### GBM
```bash
conda activate inspect_ehr
python ehr/1_csv_to_database.py       # Build FEMR database
python ehr/2_generate_labels_and_features.py  # Extract count features
python ehr/3_train_gbm.py             # Train LightGBM, save predictions
```

#### MOTOR
```bash
conda activate inspect_ehr
python ehr/motor_linear_probe_python.py   # Linear probe on MOTOR-T-Base
```

### Step 3 — CT Image Pipeline

#### 3a. Featurize (extract per-slice features, ~8-10 hrs on A100)
```bash
conda activate radfusion3
cd image/
python run_featurize_nifti.py
# Features saved to output/ct_features/features.hdf5 (~160GB)
# Supports resume if interrupted
```

#### 3b. Train classifiers (all 8 tasks, ~8-16 hrs on A100)
```bash
bash scripts/run_ct_pipeline.sh classify
# Supports resume — skips completed tasks automatically
```

#### 3c. Generate validation set predictions (needed for fusion)
```bash
bash scripts/run_ct_classify_valid.sh
```

#### 3d. Convert predictions for fusion
```bash
bash scripts/run_ct_pipeline.sh convert
```

### Step 4 — Fusion & Evaluation
```bash
conda activate base
cd fusion/
python get_model_performance.py \
    --path_to_data /path/to/data \
    --path_to_output /path/to/output
```

---

## Key Path Variables

Update these in the scripts before running:

| Variable | Description |
|----------|-------------|
| `CTPA_DIR` | Directory containing `.nii.gz` CT files |
| `COHORT_CSV` | Path to `cohort_0.2.0_master_file_anon.csv` |
| `FEMR_DB` | Path to FEMR patient database |
| `ATHENA_DIR` | Path to pruned Athena ontology |
| `MOTOR_CKPT` | Path to MOTOR-T-Base checkpoint |
| `RESNETV2_CKPT` | Path to `resnetv2_ct.ckpt` |
| `OUTPUT_DIR` | Base output directory |

---

## Results (AUROC on Test Set)

| Task | CT | MOTOR | GBM | CT+MOTOR | CT+GBM | MOTOR+GBM | CT+MOTOR+GBM |
|------|----|-------|-----|----------|--------|-----------|--------------|
| PE | 0.715 | 0.668 | 0.737 | 0.749 | 0.803 | 0.744 | **0.803** |
| 1m Mortality | 0.804 | 0.911 | 0.928 | 0.915 | 0.932 | **0.942** | 0.941 |
| 6m Mortality | 0.787 | 0.885 | 0.894 | 0.892 | 0.899 | 0.904 | **0.906** |
| 12m Mortality | 0.792 | 0.878 | 0.892 | 0.887 | 0.899 | 0.898 | **0.902** |
| 1m Readmission | 0.540 | 0.786 | 0.754 | 0.781 | 0.752 | **0.799** | 0.795 |
| 6m Readmission | 0.547 | 0.764 | 0.745 | 0.758 | 0.735 | **0.774** | 0.767 |
| 12m Readmission | 0.554 | 0.752 | 0.748 | 0.750 | 0.744 | **0.769** | 0.767 |
| 12m PH | 0.657 | 0.847 | 0.921 | 0.847 | 0.921 | **0.922** | 0.921 |

---

## Notes & Modifications from Original Code

1. **NIfTI support:** Original code used DICOM (1 file/slice). Rewrote `Dataset2D` and `run_featurize_nifti.py` to handle NIfTI volumes (all slices in one file) efficiently — loads each volume once, processes all slices in batches.

2. **Hardcoded paths:** All Stanford-specific paths updated to accept configurable paths via argparse or config files.

3. **Slice thickness dictionary:** Generated from NIfTI headers using nibabel (`data_prep/prep_image_pipeline.py`) since original `.pkl` was not publicly available.

4. **Fusion validation split:** Added `run_ct_classify_valid.sh` to generate predictions on the validation set (needed to train the fusion logistic regression).

5. **Deduplication fix:** Added deduplication by `(pid, time)` in `get_model_performance.py` to handle duplicate entries in MOTOR predictions.
