#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/score_hpt_profile.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="${ROOT_DIR}/score_hpt"

TRAIN_SET="${TRAIN_SET:-${DATASET:-maestro}}"
SEGMENT_LIST="${SEGMENT_LIST:-2 5}"
LOSS_TYPES="${LOSS_TYPES:-piano_ssm_spectral piano_ssm_spectral_plus_log_rms piano_ssm_spectral_plus_ddsp_loudness piano_ssm_combined_rm}"
DDSP_CKPTS="${DDSP_CKPTS:-}"
DDSP_PHASE="${DDSP_PHASE:-1}"
CKPT_EPOCH="${CKPT_EPOCH:-7}"
MODEL_TYPE="${MODEL_TYPE:-hpt}"

SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
BACKEND_WEIGHT="${BACKEND_WEIGHT:-${PROXY_WEIGHT:-1.0}}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

score_hpt_init_context
score_hpt_set_dataset_profile "${TRAIN_SET}"

TEST_SET="${TEST_SET:-${DEFAULT_TEST_SET}}"
EVAL_SETS="${EVAL_SETS:-${DEFAULT_EVAL_SETS}}"
DDSP_PROJECT_ROOT="${DDSP_PROJECT_ROOT:-${DDSP_PROJECT_ROOT_DEFAULT}}"
DDSP_CKPT_ROOT="${DDSP_CKPT_ROOT:-${DDSP_CKPT_ROOT_DEFAULT}}"

resolve_ddsp_ckpt() {
  score_hpt_resolve_ddsp_ckpt "${TRAIN_SET}" "$1"
}

mkdir -p "${WORKSPACE_DIR}"
read -r -a extra_args <<< "${EXTRA_OVERRIDES}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -d "${DDSP_PROJECT_ROOT}" ]; then
  echo "DDSP project root not found: ${DDSP_PROJECT_ROOT}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} not found in PATH. Please activate the correct environment first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

required_datasets=("${TRAIN_SET}" "${TEST_SET}")
while IFS= read -r dataset_name; do
  required_datasets+=("${dataset_name}")
done < <(score_hpt_collect_eval_datasets "${EVAL_SETS}")

score_hpt_prepare_required_datasets "${PYTHON_BIN}" "${required_datasets[@]}"
score_hpt_set_dataset_profile "${TRAIN_SET}"

run_one() {
  local segment="$1"
  local ckpt="$2"
  local audio_loss="$3"
  local seg_tag

  seg_tag="$(score_hpt_segment_tag "${segment}")"

  echo "============================================================"
  echo "Route III ablation"
  echo "Train set         : ${TRAIN_SET}"
  echo "Test set          : ${TEST_SET}"
  echo "Model             : ${MODEL_TYPE}"
  echo "Proxy type        : ${DIFFSYNTH_PROXY_TYPE}"
  echo "Proxy checkpoint  : ${ckpt}"
  echo "Backend seg (s)   : ${segment}"
  echo "Backend objective : ${audio_loss}"
  echo "============================================================"

  "${PYTHON_BIN}" pytorch/train_ddsp.py \
    "exp.workspace=${WORKSPACE_DIR}" \
    "dataset.train_set=${TRAIN_SET}" \
    "dataset.test_set=${TEST_SET}" \
    "dataset.eval_sets=${EVAL_SETS}" \
    "model.type=${MODEL_TYPE}" \
    "model.input2=onset" \
    "model.input3=frame" \
    "score_informed.method=note_editor" \
    "loss.supervised_weight=${SUPERVISED_WEIGHT}" \
    "loss.proxy_weight=${BACKEND_WEIGHT}" \
    "loss.velocity_prior_weight=${PRIOR_WEIGHT}" \
    "proxy.enabled=true" \
    "proxy.type=${DIFFSYNTH_PROXY_TYPE}" \
    "proxy.project_root=${DDSP_PROJECT_ROOT}" \
    "proxy.checkpoint=${ckpt}" \
    "proxy.backend_segment_seconds=${segment}" \
    "proxy.audio_loss.type=${audio_loss}" \
    "wandb.comment=${TRAIN_SET}_ddsp_${seg_tag}_${audio_loss}" \
    "${extra_args[@]}"
}

if [ -n "${DDSP_CKPTS}" ]; then
  for spec in ${DDSP_CKPTS}; do
    segment="${spec%%=*}"
    ckpt="${spec#*=}"
    for audio_loss in ${LOSS_TYPES}; do
      run_one "${segment}" "${ckpt}" "${audio_loss}"
    done
  done
  exit 0
fi

for segment in ${SEGMENT_LIST}; do
  ckpt="$(resolve_ddsp_ckpt "${segment}")"
  if [ -z "${ckpt}" ]; then
    echo "Skip ${segment}s: missing DDSP checkpoint under ${DDSP_CKPT_ROOT}" >&2
    continue
  fi
  for audio_loss in ${LOSS_TYPES}; do
    run_one "${segment}" "${ckpt}" "${audio_loss}"
  done
done
