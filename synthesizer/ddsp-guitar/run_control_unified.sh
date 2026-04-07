#!/bin/bash
set -euo pipefail

SEGMENT_SECONDS=${1:-10}
SYNTH_CKPT=${2:?Provide synthesis checkpoint path as second arg}
TRAIN_DATA=${3:-"./artefacts/guitarset_dataset_data_trn.pt"}
VAL_DATA=${4:-"./artefacts/guitarset_dataset_data_val.pt"}
GPU_ID=${GPU_ID:-0}

python train_control.py   --segment_seconds "$SEGMENT_SECONDS"   --sample_rate 22050   --frame_rate 100   --synthesis_model_checkpoint "$SYNTH_CKPT"   --train_prepared_data_path "$TRAIN_DATA"   --val_prepared_data_path "$VAL_DATA"   --gpu "$GPU_ID"   --wandb_run_name "ddsp_guitar_control_${SEGMENT_SECONDS}s"
