#!/usr/bin/env bash
set -euo pipefail

# Train the three SFProxy ablation models we actually care about:
# - coverage_v2
# - realism_v2
# - mixed_v2
#
# Data is prepared once and then reused automatically. This lets you tweak
# model/loss settings later without re-exporting synthetic data.

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

INSTRUMENT="${1:-${INSTRUMENT:-piano}}"
GUITAR_DATASET="${GUITAR_DATASET:-francoisleduc}"
SEGMENT_LIST="${SEGMENT_LIST:-2 5 10}"
BOUNDARY_MODE_LIST="${BOUNDARY_MODE_LIST:-default fixed discovered}"
TRAIN_PRESETS="${TRAIN_PRESETS:-coverage_v2 realism_v2 mixed_v2}"

TRAIN_DATASET_SIZE="${TRAIN_DATASET_SIZE:-20000}"
VAL_DATASET_SIZE="${VAL_DATASET_SIZE:-2000}"
REUSE_DATA="${REUSE_DATA:-1}"
DISCOVER_BOUNDARIES="${DISCOVER_BOUNDARIES:-1}"

MIXED_COMPONENT_NAMES="${MIXED_COMPONENT_NAMES:-boundary coverage realism stress}"
MIXED_WEIGHTS="${MIXED_WEIGHTS:-0.30 0.40 0.20 0.10}"

WANDB_PROJECT="${WANDB_PROJECT:-sfproxy_ablation}"
WANDB_GROUP="${WANDB_GROUP:-sfproxy_${INSTRUMENT}_ablation}"

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
TRAIN_PRESETS="$(normalize_list "${TRAIN_PRESETS}")"

if [[ "${MIXED_COMPONENT_NAMES}" != "boundary coverage realism stress" ]]; then
  echo "This ablation script expects the four mixed components in order: boundary coverage realism stress." >&2
  exit 1
fi

read -r -a mixed_weights <<< "${MIXED_WEIGHTS}"
if [[ "${#mixed_weights[@]}" -ne 4 ]]; then
  echo "MIXED_WEIGHTS must contain exactly four values: boundary coverage realism stress." >&2
  exit 1
fi

normalize_preset_name() {
  case "$1" in
    coverage_shared_legacy) echo "coverage_v1" ;;
    realism_shared_legacy) echo "realism_v1" ;;
    *) echo "$1" ;;
  esac
}

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
    echo "Unsupported INSTRUMENT='${INSTRUMENT}'. Expected 'piano' or 'guitar'." >&2
    exit 1
    ;;
esac

prepare_data() {
  local preset="$1"
  local segment_seconds="$2"
  local boundary_mode="$3"

  INSTRUMENT="${INSTRUMENT}" \
  GUITAR_DATASET="${GUITAR_DATASET}" \
  WORKSPACE_BASE="${WORKSPACE_BASE}" \
  ANALYSIS_DIR="${ANALYSIS_DIR}" \
  TARGET_PRESET="${preset}" \
  SEGMENT_SECONDS="${segment_seconds}" \
  BOUNDARY_MODE="${boundary_mode}" \
  TRAIN_DATASET_SIZE="${TRAIN_DATASET_SIZE}" \
  VAL_DATASET_SIZE="${VAL_DATASET_SIZE}" \
  REUSE_DATA="${REUSE_DATA}" \
  DISCOVER_BOUNDARIES="${DISCOVER_BOUNDARIES}" \
  MIXED_COMPONENT_NAMES="${MIXED_COMPONENT_NAMES}" \
  MIXED_WEIGHTS="${MIXED_WEIGHTS}" \
  EXTRA_EXPORT_OVERRIDES="${EXTRA_EXPORT_OVERRIDES}" \
  "${SCRIPT_DIR}/preprocess_sfproxy_data.sh"
}

run_train() {
  local train_tag="$1"
  local segment_seconds="$2"
  local boundary_mode="$3"

  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  python "${ROOT_DIR}/synth-proxy/src/train.py" \
    --config-name train \
    "paths.repo_root=${ROOT_DIR}" \
    "paths.workspace_dir=${WORKSPACE_BASE}" \
    "sampler_preset=${train_tag}" \
    "segment_seconds=${segment_seconds}" \
    "boundary_mode=${boundary_mode}" \
    "dataset.name=${DATASET_NAME}" \
    "dataset.train.instrument_name=${INSTRUMENT_NAME}" \
    "dataset.val.instrument_name=${INSTRUMENT_NAME}" \
    "reset_output_dir=true" \
    "${extra_train_args[@]}"
}

cd "${ROOT_DIR}"
read -r -a extra_train_args <<< "${EXTRA_TRAIN_OVERRIDES}"

for segment_seconds in ${SEGMENT_LIST}; do
  for boundary_mode in ${BOUNDARY_MODE_LIST}; do
    for raw_preset in ${TRAIN_PRESETS}; do
      preset="$(normalize_preset_name "${raw_preset}")"
      case "${preset}" in
        coverage_v2|realism_v2|mixed_v2)
          ;;
        *)
          echo "Unsupported TRAIN_PRESET='${preset}'. Use coverage_v2, realism_v2, mixed_v2." >&2
          exit 1
          ;;
      esac

      train_tag="${preset}"
      if [[ "${preset}" == "mixed_v2" ]]; then
        train_tag="$(mixed_train_tag)"
      fi

      echo "============================================================"
      echo "Train preset     : ${preset}"
      echo "Train tag        : ${train_tag}"
      echo "Instrument       : ${INSTRUMENT_NAME}"
      echo "Segment seconds  : ${segment_seconds}"
      echo "Boundary mode    : ${boundary_mode}"
      if [[ "${preset}" == "mixed_v2" ]]; then
        echo "Mixed weights    : ${MIXED_WEIGHTS}"
      fi
      echo "W&B project      : ${WANDB_PROJECT}"
      echo "W&B group        : ${WANDB_GROUP}"
      echo "============================================================"

      prepare_data "${preset}" "${segment_seconds}" "${boundary_mode}"
      run_train "${train_tag}" "${segment_seconds}" "${boundary_mode}"
    done
  done
done
