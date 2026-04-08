#!/usr/bin/env bash
set -euo pipefail

# Single SFProxy v2 training run.
# Use this after you want to tweak model/loss settings while reusing exported data.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MACHINE="${MACHINE:-5090}"
case "${MACHINE}" in
  3090)
    DEFAULT_WORKSPACE_BASE="${ROOT_DIR}/_runs"
    DEFAULT_ANALYSIS_DIR="${ROOT_DIR}/data_analysis"
    ;;
  5090)
    DEFAULT_WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data"
    DEFAULT_ANALYSIS_DIR="${ROOT_DIR}/data_analysis"
    ;;
  *)
    echo "Unsupported MACHINE='${MACHINE}'. Expected '3090' or '5090'." >&2
    exit 1
    ;;
esac

WORKSPACE_BASE="${WORKSPACE_BASE:-${DEFAULT_WORKSPACE_BASE}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${DEFAULT_ANALYSIS_DIR}}"

INSTRUMENT="${INSTRUMENT:-piano}"
GUITAR_DATASET="${GUITAR_DATASET:-francoisleduc}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-2}"
BOUNDARY_MODE="${BOUNDARY_MODE:-discovered}"
TRAIN_PRESET="${TRAIN_PRESET:-mixed_v2}"

TRAIN_DATASET_SIZE="${TRAIN_DATASET_SIZE:-20000}"
VAL_DATASET_SIZE="${VAL_DATASET_SIZE:-2000}"
REUSE_DATA="${REUSE_DATA:-1}"
DISCOVER_BOUNDARIES="${DISCOVER_BOUNDARIES:-1}"

MIXED_COMPONENT_NAMES="${MIXED_COMPONENT_NAMES:-boundary coverage realism stress}"
MIXED_WEIGHTS="${MIXED_WEIGHTS:-0.30 0.40 0.20 0.10}"

WANDB_PROJECT="${WANDB_PROJECT:-sfproxy_ablation}"
WANDB_GROUP="${WANDB_GROUP:-sfproxy_${INSTRUMENT}_single}"

EXTRA_EXPORT_OVERRIDES="${EXTRA_EXPORT_OVERRIDES:-}"
EXTRA_TRAIN_OVERRIDES="${EXTRA_TRAIN_OVERRIDES:-}"

normalize_list() {
  local raw="$1"
  raw="${raw//,/ }"
  local -a items=()
  read -r -a items <<< "${raw}"
  echo "${items[*]}"
}

MIXED_COMPONENT_NAMES="$(normalize_list "${MIXED_COMPONENT_NAMES}")"
MIXED_WEIGHTS="$(normalize_list "${MIXED_WEIGHTS}")"

read -r -a mixed_weights <<< "${MIXED_WEIGHTS}"
if [[ "${MIXED_COMPONENT_NAMES}" != "boundary coverage realism stress" || "${#mixed_weights[@]}" -ne 4 ]]; then
  echo "This script expects four mixed components in order: boundary coverage realism stress." >&2
  exit 1
fi

sanitize_weight_token() {
  local token="$1"
  token="${token//./p}"
  token="${token//-/_}"
  echo "${token}"
}

mixed_train_tag() {
  if [[ "${MIXED_WEIGHTS}" == "0.30 0.40 0.20 0.10" ]]; then
    echo "mixed_v2"
    return 0
  fi
  echo "mixed_v2_b$(sanitize_weight_token "${mixed_weights[0]}")_c$(sanitize_weight_token "${mixed_weights[1]}")_r$(sanitize_weight_token "${mixed_weights[2]}")_s$(sanitize_weight_token "${mixed_weights[3]}")"
}

case "${INSTRUMENT}" in
  piano)
    INSTRUMENT_NAME="salamander_piano"
    DATASET_NAME="piano"
    ;;
  guitar)
    case "${GUITAR_DATASET}" in
      francoisleduc)
        INSTRUMENT_NAME="guitar"
        DATASET_NAME="guitar"
        ;;
      gaps)
        INSTRUMENT_NAME="guitar_gaps"
        DATASET_NAME="guitar_gaps"
        ;;
      *)
        echo "Unsupported GUITAR_DATASET='${GUITAR_DATASET}'." >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported INSTRUMENT='${INSTRUMENT}'." >&2
    exit 1
    ;;
esac

case "${TRAIN_PRESET}" in
  coverage_v2|realism_v2)
    TRAIN_TAG="${TRAIN_PRESET}"
    PREPARE_TARGET="${TRAIN_PRESET}"
    ;;
  mixed_v2)
    TRAIN_TAG="$(mixed_train_tag)"
    PREPARE_TARGET="mixed_v2"
    ;;
  *)
    echo "Unsupported TRAIN_PRESET='${TRAIN_PRESET}'. Use coverage_v2, realism_v2, or mixed_v2." >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "Train preset     : ${TRAIN_PRESET}"
echo "Train tag        : ${TRAIN_TAG}"
echo "Instrument       : ${INSTRUMENT_NAME}"
echo "Segment seconds  : ${SEGMENT_SECONDS}"
echo "Boundary mode    : ${BOUNDARY_MODE}"
if [[ "${TRAIN_PRESET}" == "mixed_v2" ]]; then
  echo "Mixed weights    : ${MIXED_WEIGHTS}"
fi
echo "============================================================"

INSTRUMENT="${INSTRUMENT}" \
GUITAR_DATASET="${GUITAR_DATASET}" \
WORKSPACE_BASE="${WORKSPACE_BASE}" \
ANALYSIS_DIR="${ANALYSIS_DIR}" \
TARGET_PRESET="${PREPARE_TARGET}" \
SEGMENT_SECONDS="${SEGMENT_SECONDS}" \
BOUNDARY_MODE="${BOUNDARY_MODE}" \
TRAIN_DATASET_SIZE="${TRAIN_DATASET_SIZE}" \
VAL_DATASET_SIZE="${VAL_DATASET_SIZE}" \
REUSE_DATA="${REUSE_DATA}" \
DISCOVER_BOUNDARIES="${DISCOVER_BOUNDARIES}" \
MIXED_COMPONENT_NAMES="${MIXED_COMPONENT_NAMES}" \
MIXED_WEIGHTS="${MIXED_WEIGHTS}" \
EXTRA_EXPORT_OVERRIDES="${EXTRA_EXPORT_OVERRIDES}" \
"${SCRIPT_DIR}/preprocess_sfproxy_data.sh"

cd "${ROOT_DIR}"
read -r -a extra_train_args <<< "${EXTRA_TRAIN_OVERRIDES}"

WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_GROUP="${WANDB_GROUP}" \
python "${ROOT_DIR}/synth-proxy/src/train.py" \
  --config-name train \
  "paths.repo_root=${ROOT_DIR}" \
  "paths.workspace_dir=${WORKSPACE_BASE}" \
  "sampler_preset=${TRAIN_TAG}" \
  "segment_seconds=${SEGMENT_SECONDS}" \
  "boundary_mode=${BOUNDARY_MODE}" \
  "dataset.name=${DATASET_NAME}" \
  "dataset.train.instrument_name=${INSTRUMENT_NAME}" \
  "dataset.val.instrument_name=${INSTRUMENT_NAME}" \
  "reset_output_dir=true" \
  "${extra_train_args[@]}"
