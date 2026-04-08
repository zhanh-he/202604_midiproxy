#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/sfproxy_profile.sh"

INSTRUMENT="${INSTRUMENT:-piano}"
PIANO_DATASET="${PIANO_DATASET:-maestro}"
GUITAR_DATASET="${GUITAR_DATASET:-francoisleduc}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-2}"
BOUNDARY_MODE="${BOUNDARY_MODE:-discovered}"
TRAIN_PRESET="${TRAIN_PRESET:-mixed_v2}"

WANDB_PROJECT="${WANDB_PROJECT:-sfproxy_ablation}"
WANDB_GROUP="${WANDB_GROUP:-sfproxy_${INSTRUMENT}_single}"
EXTRA_TRAIN_OVERRIDES="${EXTRA_TRAIN_OVERRIDES:-}"

case "${TRAIN_PRESET}" in
  boundary_v2|coverage_v2|realism_v2|stress_v2|mixed_v2)
    ;;
  *)
    echo "Unsupported TRAIN_PRESET='${TRAIN_PRESET}'." >&2
    exit 1
    ;;
esac

sfproxy_set_profile "${INSTRUMENT}" "${PIANO_DATASET}" "${GUITAR_DATASET}"

INSTRUMENT="${INSTRUMENT}" \
PIANO_DATASET="${PIANO_DATASET}" \
GUITAR_DATASET="${GUITAR_DATASET}" \
TARGET_PRESET="${TRAIN_PRESET}" \
SEGMENT_SECONDS="${SEGMENT_SECONDS}" \
BOUNDARY_MODE="${BOUNDARY_MODE}" \
"${SCRIPT_DIR}/preprocess_sfproxy_data.sh"

cd "${ROOT_DIR}"
read -r -a extra_train_args <<< "${EXTRA_TRAIN_OVERRIDES}"

echo "train ${TRAIN_PRESET} ${INSTRUMENT_NAME} ${SEGMENT_SECONDS}s ${BOUNDARY_MODE}"

WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_GROUP="${WANDB_GROUP}" \
python "${ROOT_DIR}/synth-proxy/src/train.py" \
  --config-name train \
  "paths.repo_root=${ROOT_DIR}" \
  "sampler_preset=${TRAIN_PRESET}" \
  "segment_seconds=${SEGMENT_SECONDS}" \
  "boundary_mode=${BOUNDARY_MODE}" \
  "dataset.name=${DATASET_NAME}" \
  "dataset.train.instrument_name=${INSTRUMENT_NAME}" \
  "dataset.val.instrument_name=${INSTRUMENT_NAME}" \
  "reset_output_dir=true" \
  "${extra_train_args[@]}"
