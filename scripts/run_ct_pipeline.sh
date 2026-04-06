#!/bin/bash
# CT Image Pipeline for INSPECT Benchmark
# Usage: bash run_ct_pipeline.sh [stage]
#   stage: featurize | classify | convert | all (default: all)

set -e

PYTHON=/home/pkrish52/miniconda3/envs/radfusion3/bin/python
IMAGE_DIR=/scratch/pkrish52/INSPECT/repos/INSPECT_public/image
OUTPUT_DIR=/scratch/pkrish52/INSPECT/output

STAGE=${1:-all}

###############################################################################
# Stage 1: Featurize (extract 2D slice features from all CT images)
###############################################################################
if [[ "$STAGE" == "featurize" || "$STAGE" == "all" ]]; then
    echo "=========================================="
    echo "Stage 1: Featurizing CT images..."
    echo "=========================================="
    cd $IMAGE_DIR

    $PYTHON run_featurize.py model=resnetv2_ct \
        dataset=stanford \
        dataset.transform.final_size=224 \
        dataset.batch_size=64 \
        dataset.transform.channels=window \
        trainer.num_workers=8

    echo "Converting features to HDF5..."
    $PYTHON convert_to_hdf5.py

    echo "Stage 1 complete. Features at: $OUTPUT_DIR/ct_features/features.hdf5"
fi

###############################################################################
# Stage 2: Classification (train 1D sequence encoder for each task)
###############################################################################
if [[ "$STAGE" == "classify" || "$STAGE" == "all" ]]; then
    echo "=========================================="
    echo "Stage 2: Training classifiers..."
    echo "=========================================="
    cd $IMAGE_DIR

    # Helper: skip task if test_preds.csv already exists for this target
    skip_if_done() {
        local target=$1
        local existing=$(find $OUTPUT_DIR/ct_classify_exp -name "test_preds.csv" -path "*${target}*" 2>/dev/null | head -1)
        if [[ -n "$existing" ]]; then
            echo "--- SKIPPING ${target} (already done: $existing) ---"
            return 0
        fi
        return 1
    }

    # Common args
    COMMON="model=model_1d dataset=stanford_featurized \
        dataset.pretrain_args.model_type=resnetv2_101_ct \
        dataset.pretrain_args.channel_type=window \
        dataset.feature_size=768 \
        dataset.num_slices=250 \
        dataset.weighted_sample=true \
        trainer.max_epochs=50 \
        lr=0.001 \
        trainer.seed=0 \
        n_gpus=1 \
        trainer.strategy=auto \
        dataset.batch_size=128 \
        trainer.num_workers=4 \
        dataset.num_slices=250"

    echo "--- PE ---"
    skip_if_done "pe_positive_nlp" || $PYTHON run_classify.py $COMMON \
        dataset.target=pe_positive_nlp \
        model.aggregation=max \
        model.seq_encoder.rnn_type=LSTM \
        model.seq_encoder.bidirectional=true \
        model.seq_encoder.num_layers=1 \
        model.seq_encoder.hidden_size=128 \
        model.seq_encoder.dropout_prob=0.5

    echo "--- 1mo Mortality ---"
    skip_if_done "1_month_mortality" || $PYTHON run_classify.py $COMMON \
        dataset.target=1_month_mortality \
        model.aggregation=max \
        model.seq_encoder.rnn_type=GRU \
        model.seq_encoder.bidirectional=true \
        model.seq_encoder.num_layers=1 \
        model.seq_encoder.hidden_size=128 \
        model.seq_encoder.dropout_prob=0.25

    echo "--- 6mo Mortality ---"
    skip_if_done "6_month_mortality" || $PYTHON run_classify.py $COMMON \
        dataset.target=6_month_mortality \
        model.aggregation=mean \
        model.seq_encoder.rnn_type=GRU \
        model.seq_encoder.bidirectional=true \
        model.seq_encoder.num_layers=1 \
        model.seq_encoder.hidden_size=128 \
        model.seq_encoder.dropout_prob=0.0

    echo "--- 12mo Mortality ---"
    skip_if_done "12_month_mortality" || $PYTHON run_classify.py $COMMON \
        dataset.target=12_month_mortality \
        model.aggregation=attention \
        model.seq_encoder.rnn_type=GRU \
        model.seq_encoder.bidirectional=true \
        model.seq_encoder.num_layers=1 \
        model.seq_encoder.hidden_size=128 \
        model.seq_encoder.dropout_prob=0.5

    echo "--- 1mo Readmission ---"
    skip_if_done "1_month_readmission" || $PYTHON run_classify.py $COMMON \
        dataset.target=1_month_readmission \
        model.aggregation=max \
        model.seq_encoder.rnn_type=LSTM \
        model.seq_encoder.bidirectional=true \
        model.seq_encoder.num_layers=1 \
        model.seq_encoder.hidden_size=128 \
        model.seq_encoder.dropout_prob=0.0

    echo "--- 6mo Readmission ---"
    skip_if_done "6_month_readmission" || $PYTHON run_classify.py $COMMON \
        dataset.target=6_month_readmission \
        model.aggregation=max \
        model.seq_encoder.rnn_type=LSTM \
        model.seq_encoder.bidirectional=true \
        model.seq_encoder.num_layers=1 \
        model.seq_encoder.hidden_size=128 \
        model.seq_encoder.dropout_prob=0.5

    echo "--- 12mo Readmission ---"
    skip_if_done "12_month_readmission" || $PYTHON run_classify.py $COMMON \
        dataset.target=12_month_readmission \
        model.aggregation=mean \
        model.seq_encoder.rnn_type=LSTM \
        model.seq_encoder.bidirectional=true \
        model.seq_encoder.num_layers=1 \
        model.seq_encoder.hidden_size=128 \
        model.seq_encoder.dropout_prob=0.25

    echo "--- 12mo PH ---"
    skip_if_done "12_month_PH" || $PYTHON run_classify.py $COMMON \
        dataset.target=12_month_PH \
        model.aggregation=attention \
        model.seq_encoder.rnn_type=GRU \
        model.seq_encoder.bidirectional=true \
        model.seq_encoder.num_layers=1 \
        model.seq_encoder.hidden_size=128 \
        model.seq_encoder.dropout_prob=0.25

    echo "Stage 2 complete."
fi

###############################################################################
# Stage 3: Convert predictions for fusion
###############################################################################
if [[ "$STAGE" == "convert" || "$STAGE" == "all" ]]; then
    echo "=========================================="
    echo "Stage 3: Converting predictions for fusion..."
    echo "=========================================="
    $PYTHON /scratch/pkrish52/INSPECT/convert_ct_preds_for_fusion.py
    echo "Stage 3 complete."
fi

echo "=========================================="
echo "CT Pipeline complete!"
echo "=========================================="
