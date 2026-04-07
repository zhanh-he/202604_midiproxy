#!/usr/bin/env bash
set -euo pipefail

# Route IV / SFProxy adaptation ablations inside Score-HPT.
# This script assumes the HDF5 datasets already exist under exp.workspace/hdf5s.
# Example:
#   PROXY_CKPT=/path/to/sfproxy.ckpt bash run_scripts/train_route4_ablation.sh
#   SEGMENT_LIST="2 5 10" LOSS_TYPES="smooth_l1 l1" PROXY_CKPT=/path/to/sfproxy.ckpt bash run_scripts/train_route4_ablation.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

WORKSPACE_DIR="${WORKSPACE_DIR:-${ROOT_DIR}/score_hpt/workspaces}"
PROJECT_ROOT="${PROJECT_ROOT:-${ROOT_DIR}/synth-proxy}"
PROXY_CKPT="${PROXY_CKPT:-${1:-}}"
INSTRUMENT_NAME="${INSTRUMENT_NAME:-salamander_piano}"
MODEL_TYPE="${MODEL_TYPE:-hpt}"
SCORE_METHOD="${SCORE_METHOD:-note_editor}"
INPUT2="${INPUT2:-onset}"
INPUT3="${INPUT3:-frame}"
SUPERVISED_LOSS="${SUPERVISED_LOSS:-kim_bce_l1}"
PROXY_WEIGHT="${PROXY_WEIGHT:-1.0}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-1.0}"
TOTAL_ITER="${TOTAL_ITER:-120000}"
SAVE_ITER="${SAVE_ITER:-10000}"
EVAL_ITER="${EVAL_ITER:-10000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-12}"
WANDB_PROJECT="${WANDB_PROJECT:-202602_smc}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

SEGMENT_LIST="${SEGMENT_LIST:-2 5 10}"
CROP_MODES="${CROP_MODES:-random}"
LOSS_TYPES="${LOSS_TYPES:-smooth_l1}"
PRIOR_WEIGHTS="${PRIOR_WEIGHTS:-0.0 0.01}"
WARMUP_ITERS="${WARMUP_ITERS:-0}"

if [[ -z "${PROXY_CKPT}" || ! -f "${PROXY_CKPT}" ]]; then
  echo "Set PROXY_CKPT to a trained SFProxy Lightning checkpoint (.ckpt)." >&2
  exit 1
fi

mkdir -p "${WORKSPACE_DIR}"
read -r -a extra_args <<< "${EXTRA_OVERRIDES}"

cd "${ROOT_DIR}/score_hpt"

for SEGMENT_SECONDS in ${SEGMENT_LIST}; do
  SEGMENT_TAG="${SEGMENT_SECONDS%.0}"
  for CROP_MODE in ${CROP_MODES}; do
    for PROXY_LOSS in ${LOSS_TYPES}; do
      for PRIOR_WEIGHT in ${PRIOR_WEIGHTS}; do
        TAG="route4_${INSTRUMENT_NAME}_${SEGMENT_TAG}s_${CROP_MODE}_${PROXY_LOSS}_prior${PRIOR_WEIGHT}"
        echo "============================================================"
        echo "Route IV run      : ${TAG}"
        echo "Proxy checkpoint  : ${PROXY_CKPT}"
        echo "Workspace         : ${WORKSPACE_DIR}"
        echo "Front-end seg (s) : 10"
        echo "Backend crop (s)  : ${SEGMENT_SECONDS}"
        echo "Crop mode         : ${CROP_MODE}"
        echo "Proxy loss        : ${PROXY_LOSS}"
        echo "Prior weight      : ${PRIOR_WEIGHT}"
        echo "============================================================"

        "${PYTHON_BIN}" pytorch/train_proxy.py \
          "exp.workspace=${WORKSPACE_DIR}" \
          "exp.total_iteration=${TOTAL_ITER}" \
          "exp.save_iteration=${SAVE_ITER}" \
          "exp.eval_iteration=${EVAL_ITER}" \
          "exp.batch_size=${BATCH_SIZE}" \
          "feature.segment_seconds=10.0" \
          "exp.num_workers=${NUM_WORKERS}" \
          "wandb.project=${WANDB_PROJECT}" \
          "wandb.name=${TAG}" \
          "wandb.comment=${TAG}" \
          "model.type=${MODEL_TYPE}" \
          "model.input2=${INPUT2}" \
          "model.input3=${INPUT3}" \
          "score_informed.method=${SCORE_METHOD}" \
          "loss.loss_type=${SUPERVISED_LOSS}" \
          "loss.supervised_weight=${SUPERVISED_WEIGHT}" \
          "loss.proxy_weight=${PROXY_WEIGHT}" \
          "loss.velocity_prior_weight=${PRIOR_WEIGHT}" \
          "proxy.enabled=true" \
          "proxy.type=diffproxy" \
          "proxy.project_root=${PROJECT_ROOT}" \
          "proxy.checkpoint=${PROXY_CKPT}" \
          "proxy.backend_segment_seconds=${SEGMENT_SECONDS}" \
          "proxy.crop_mode=${CROP_MODE}" \
          "proxy.warmup_iterations=${WARMUP_ITERS}" \
          "proxy.supervision.hop_size=221" \
          "proxy.sfproxy.instrument_name=${INSTRUMENT_NAME}" \
          "proxy.sfproxy.loss_type=${PROXY_LOSS}" \
          "proxy.sfproxy.use_gt_aligned_note_events=true" \
          "proxy.sfproxy.feature.hop=221" \
          "${extra_args[@]}"
      done
    done
  done
done
