#!/bin/bash
set -euo pipefail

SEGMENT_SECONDS=${1:-10}
TRAIN_DATA=${2:-"./artefacts/guitarset_dataset_data_trn.pt"}
VAL_DATA=${3:-"./artefacts/guitarset_dataset_data_val.pt"}
GPU_ID=${GPU_ID:-0}

python train_synthesis.py   --segment_seconds "$SEGMENT_SECONDS"   --sample_rate 22050   --frame_rate 100   --train_prepared_data_path "$TRAIN_DATA"   --val_prepared_data_path "$VAL_DATA"   --gpu "$GPU_ID"   --wandb_run_name "ddsp_guitar_synth_${SEGMENT_SECONDS}s"
