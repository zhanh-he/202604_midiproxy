#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/score_hpt_profile.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="${ROOT_DIR}/score_hpt"

TRAIN_SET="${TRAIN_SET:-${GUITAR_DATASET:-francoisleduc}}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-5}"

BATCH_SIZE="${BATCH_SIZE:-4}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
PROXY_WEIGHT="${PROXY_WEIGHT:-1.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.0}"
BACKEND_SEGMENT_SECONDS="${BACKEND_SEGMENT_SECONDS:-${SEGMENT_SECONDS}}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

score_hpt_init_context
score_hpt_set_dataset_profile "${TRAIN_SET}"

TEST_SET="${TEST_SET:-${DEFAULT_TEST_SET}}"
EVAL_SETS="${EVAL_SETS:-${DEFAULT_EVAL_SETS}}"
DDSP_PROJECT_ROOT="${DDSP_PROJECT_ROOT:-${DDSP_PROJECT_ROOT_DEFAULT}}"
DDSP_CKPT_ROOT="${DDSP_CKPT_ROOT:-${DDSP_CKPT_ROOT_DEFAULT}}"
DDSP_CKPT="${DDSP_CKPT:-$(score_hpt_resolve_ddsp_ckpt "${TRAIN_SET}" "${SEGMENT_SECONDS}")}"

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

if [ ! -f "${DDSP_CKPT}" ]; then
  echo "DDSP checkpoint not found: ${DDSP_CKPT}" >&2
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

"${PYTHON_BIN}" pytorch/train_ddsp.py \
  "exp.workspace=${WORKSPACE_DIR}" \
  "exp.batch_size=${BATCH_SIZE}" \
  "dataset.train_set=${TRAIN_SET}" \
  "dataset.test_set=${TEST_SET}" \
  "dataset.eval_sets=${EVAL_SETS}" \
  "model.type=hpt" \
  "model.input2=onset" \
  "model.input3=frame" \
  "score_informed.method=note_editor" \
  "loss.supervised_weight=${SUPERVISED_WEIGHT}" \
  "loss.proxy_weight=${PROXY_WEIGHT}" \
  "loss.velocity_prior_weight=${PRIOR_WEIGHT}" \
  "proxy.enabled=true" \
  "proxy.type=${DIFFSYNTH_PROXY_TYPE}" \
  "proxy.project_root=${DDSP_PROJECT_ROOT}" \
  "proxy.checkpoint=${DDSP_CKPT}" \
  "proxy.backend_segment_seconds=${BACKEND_SEGMENT_SECONDS}" \
  "wandb.comment=${TRAIN_SET}" \
  "${extra_args[@]}"
