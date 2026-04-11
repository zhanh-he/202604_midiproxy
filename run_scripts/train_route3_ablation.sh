#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/score_hpt_profile.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="${ROOT_DIR}/score_hpt"

TRAIN_SET="${TRAIN_SET:-maestro}"
SEGMENT_LIST="${SEGMENT_LIST:-2 5}"
LOSS_TYPES="${LOSS_TYPES:-piano_ssm_spectral piano_ssm_spectral_plus_log_rms piano_ssm_spectral_plus_ddsp_loudness piano_ssm_combined_rm}"
MODEL_TYPE="${MODEL_TYPE:-hpt}"
SCORE_METHOD="${SCORE_METHOD:-note_editor}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"

SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
BACKEND_WEIGHT="${BACKEND_WEIGHT:-1.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

score_hpt_init_context
score_hpt_set_dataset_profile "${TRAIN_SET}"

mkdir -p "${WORKSPACE_DIR}"
read -r -a extra_args <<< "${EXTRA_OVERRIDES}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -d "${DDSP_PROJECT_ROOT_DEFAULT}" ]; then
  echo "DDSP project root not found: ${DDSP_PROJECT_ROOT_DEFAULT}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} not found in PATH. Please activate the correct environment first." >&2
  exit 1
fi

if [ "${MODEL_TYPE}" != "hpt" ] && [ "${MODEL_TYPE}" != "filmunet" ]; then
  echo "Unsupported MODEL_TYPE: ${MODEL_TYPE}. Use hpt or filmunet." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

required_datasets=("${TRAIN_SET}" "${DEFAULT_TEST_SET}")
while IFS= read -r dataset_name; do
  required_datasets+=("${dataset_name}")
done < <(score_hpt_collect_eval_datasets "${DEFAULT_EVAL_SETS}")

score_hpt_prepare_required_datasets "${PYTHON_BIN}" "${required_datasets[@]}"
score_hpt_set_dataset_profile "${TRAIN_SET}"

run_one() {
  local segment="$1"
  local ckpt="$2"
  local audio_loss="$3"
  local score_method
  local model_input2

  score_method="${SCORE_METHOD}"
  if [ "${MODEL_TYPE}" = "filmunet" ]; then
    score_method="direct"
  elif [ "${score_method}" != "direct" ] && [ "${score_method}" != "note_editor" ]; then
    echo "Unsupported SCORE_METHOD for ${MODEL_TYPE}: ${score_method}" >&2
    exit 1
  fi
  model_input2="null"
  if [ "${MODEL_TYPE}" = "hpt" ] && [ "${score_method}" = "note_editor" ]; then
    model_input2="onset"
  fi
  local pretrained_args=()
  if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    pretrained_args+=("model.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}")
  fi

  echo "============================================================"
  echo "Route III ablation"
  echo "Train set         : ${TRAIN_SET}"
  echo "Test set          : ${DEFAULT_TEST_SET}"
  echo "Model             : ${MODEL_TYPE}"
  echo "Score method      : ${score_method}"
  echo "Proxy type        : ${DIFFSYNTH_PROXY_TYPE}"
  echo "Proxy checkpoint  : ${ckpt}"
  echo "Backend seg (s)   : ${segment}"
  echo "Backend objective : ${audio_loss}"
  echo "============================================================"

  "${PYTHON_BIN}" pytorch/train_ddsp.py \
    "exp.workspace=${WORKSPACE_DIR}" \
    "dataset.train_set=${TRAIN_SET}" \
    "dataset.test_set=${DEFAULT_TEST_SET}" \
    "dataset.eval_sets=${DEFAULT_EVAL_SETS}" \
    "model.type=${MODEL_TYPE}" \
    "model.input2=${model_input2}" \
    "score_informed.method=${score_method}" \
    "loss.supervised_weight=${SUPERVISED_WEIGHT}" \
    "loss.proxy_weight=${BACKEND_WEIGHT}" \
    "loss.velocity_prior_weight=${PRIOR_WEIGHT}" \
    "proxy.enabled=true" \
    "proxy.type=${DIFFSYNTH_PROXY_TYPE}" \
    "proxy.project_root=${DDSP_PROJECT_ROOT_DEFAULT}" \
    "proxy.checkpoint=${ckpt}" \
    "proxy.backend_segment_seconds=${segment}" \
    "proxy.audio_loss.type=${audio_loss}" \
    "${pretrained_args[@]}" \
    "${extra_args[@]}"
}

for segment in ${SEGMENT_LIST}; do
  ckpt="$(score_hpt_resolve_ddsp_ckpt "${TRAIN_SET}" "${segment}")"
  if [ -z "${ckpt}" ]; then
    echo "Skip ${segment}s: missing DDSP checkpoint under ${DDSP_CKPT_ROOT_DEFAULT}" >&2
    continue
  fi
  for audio_loss in ${LOSS_TYPES}; do
    run_one "${segment}" "${ckpt}" "${audio_loss}"
  done
done
