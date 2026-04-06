#!/bin/bash
set -e

PYTHON=/home/pkrish52/miniconda3/envs/inspect_ehr/bin/python
COHORT=/scratch/pkrish52/INSPECT/data/cohort_0.2.0_master_file_anon.csv
DATABASE=/scratch/pkrish52/INSPECT/output/inspect_femr_extract/extract
OUTPUT=/scratch/pkrish52/INSPECT/output/labels_and_features
EHR_DIR=/scratch/pkrish52/INSPECT/repos/INSPECT_public/ehr

TASKS=("PE" "1_month_mortality" "6_month_mortality" "12_month_mortality" "1_month_readmission" "6_month_readmission" "12_month_readmission" "12_month_PH")

echo "=== Step 1: Generate labels and features for all 8 tasks ==="
for task in "${TASKS[@]}"; do
    echo "--- $task ---"
    mkdir -p "$OUTPUT/$task"
    cd "$EHR_DIR"
    $PYTHON 2_generate_labels_and_features.py \
        --path_to_cohort "$COHORT" \
        --path_to_database "$DATABASE" \
        --path_to_output_dir "$OUTPUT/$task" \
        --labeling_function "$task" \
        --num_threads 4
    echo "$task done"
done

echo ""
echo "=== Step 2: Train GBM for all 8 tasks ==="
for task in "${TASKS[@]}"; do
    echo "--- GBM: $task ---"
    mkdir -p /scratch/pkrish52/INSPECT/output/gbm_model_results/$task
    cd "$EHR_DIR"
    $PYTHON 3_train_gbm.py \
        --path_to_cohort "$COHORT" \
        --path_to_database "$DATABASE" \
        --path_to_label_features "$OUTPUT/$task" \
        --path_to_output_dir /scratch/pkrish52/INSPECT/output/gbm_model_results/$task
    echo "GBM $task done"
done

echo ""
echo "=== Step 3: MOTOR linear probes for all 8 tasks ==="
for task in "${TASKS[@]}"; do
    echo "--- MOTOR: $task ---"
    # Clear old repr cache
    rm -f /scratch/pkrish52/INSPECT/output/motor_results/$task/reprs_cache.pkl
    mkdir -p /scratch/pkrish52/INSPECT/output/motor_results/$task
    cd /scratch/pkrish52/INSPECT
    $PYTHON motor_linear_probe_python.py \
        --task "$task" \
        --data_path "$DATABASE" \
        --model_dir /scratch/pkrish52/INSPECT/motor_model/model \
        --dictionary_path /scratch/pkrish52/INSPECT/motor_model/dictionary \
        --labels_dir "$OUTPUT" \
        --cohort_path /scratch/pkrish52/INSPECT/data/cohort_motor.csv \
        --output_dir /scratch/pkrish52/INSPECT/output/motor_results/"$task"
    echo "MOTOR $task done"
done

echo ""
echo "=== Step 4: Fusion ==="
cd /scratch/pkrish52/INSPECT/repos/INSPECT_public
$PYTHON get_model_performance.py

echo ""
echo "=== ALL DONE ==="
