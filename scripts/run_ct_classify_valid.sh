#!/bin/bash
# Run classifiers on validation split using trained checkpoints
# Generates valid set predictions needed for fusion LR training

set -e

PYTHON=/home/pkrish52/miniconda3/envs/radfusion3/bin/python
IMAGE_DIR=/scratch/pkrish52/INSPECT/repos/INSPECT_public/image
EXP_BASE=/scratch/pkrish52/INSPECT/output/ct_classify_exp
CKPT_FILE=/tmp/inspect_ckpt_path.txt

cd $IMAGE_DIR

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
    dataset.num_slices=250 \
    test_split=valid \
    ckpt=test_only"

run_task() {
    local target=$1
    shift

    # Skip if valid preds already exist for this task
    local existing=$(find $EXP_BASE -name "test_preds.csv" -path "*${target}*valid*" 2>/dev/null | head -1)
    if [[ -n "$existing" ]]; then
        echo "--- SKIPPING ${target} (already done) ---"
        return 0
    fi

    # Find best checkpoint from training run
    local ckpt=$(find $EXP_BASE -name "*.ckpt" -path "*${target}*" ! -name "last.ckpt" 2>/dev/null | sort | tail -1)
    if [[ -z "$ckpt" ]]; then
        ckpt=$(find $EXP_BASE -name "last.ckpt" -path "*${target}*" 2>/dev/null | head -1)
    fi

    if [[ -z "$ckpt" ]]; then
        echo "--- ERROR: No checkpoint found for ${target}, skipping ---"
        return 1
    fi

    echo "--- ${target} ---"
    echo "    ckpt: $ckpt"

    # Write ckpt path to file, then patch it into the lightning model via env var
    # Run test-only using the trained checkpoint loaded at model build time
    INSPECT_CKPT_PATH="$ckpt" $PYTHON run_classify.py $COMMON \
        dataset.target=$target "$@"
}

run_task pe_positive_nlp \
    model.aggregation=max \
    model.seq_encoder.rnn_type=LSTM model.seq_encoder.bidirectional=true \
    model.seq_encoder.num_layers=1 model.seq_encoder.hidden_size=128 \
    model.seq_encoder.dropout_prob=0.5

run_task 1_month_mortality \
    model.aggregation=max \
    model.seq_encoder.rnn_type=GRU model.seq_encoder.bidirectional=true \
    model.seq_encoder.num_layers=1 model.seq_encoder.hidden_size=128 \
    model.seq_encoder.dropout_prob=0.25

run_task 6_month_mortality \
    model.aggregation=mean \
    model.seq_encoder.rnn_type=GRU model.seq_encoder.bidirectional=true \
    model.seq_encoder.num_layers=1 model.seq_encoder.hidden_size=128 \
    model.seq_encoder.dropout_prob=0.0

run_task 12_month_mortality \
    model.aggregation=attention \
    model.seq_encoder.rnn_type=GRU model.seq_encoder.bidirectional=true \
    model.seq_encoder.num_layers=1 model.seq_encoder.hidden_size=128 \
    model.seq_encoder.dropout_prob=0.5

run_task 1_month_readmission \
    model.aggregation=max \
    model.seq_encoder.rnn_type=LSTM model.seq_encoder.bidirectional=true \
    model.seq_encoder.num_layers=1 model.seq_encoder.hidden_size=128 \
    model.seq_encoder.dropout_prob=0.0

run_task 6_month_readmission \
    model.aggregation=max \
    model.seq_encoder.rnn_type=LSTM model.seq_encoder.bidirectional=true \
    model.seq_encoder.num_layers=1 model.seq_encoder.hidden_size=128 \
    model.seq_encoder.dropout_prob=0.5

run_task 12_month_readmission \
    model.aggregation=mean \
    model.seq_encoder.rnn_type=LSTM model.seq_encoder.bidirectional=true \
    model.seq_encoder.num_layers=1 model.seq_encoder.hidden_size=128 \
    model.seq_encoder.dropout_prob=0.25

run_task 12_month_PH \
    model.aggregation=attention \
    model.seq_encoder.rnn_type=GRU model.seq_encoder.bidirectional=true \
    model.seq_encoder.num_layers=1 model.seq_encoder.hidden_size=128 \
    model.seq_encoder.dropout_prob=0.25

echo "Done! Now re-run convert stage to merge valid+test predictions."
