#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="${ROOT_DIR}/score_hpt"
DDSP_PROJECT_ROOT="${DDSP_PROJECT_ROOT:-${ROOT_DIR}/synthesizer/ddsp-piano-pytorch}"

WORKSPACE_BASE="${WORKSPACE_BASE:-/media/mengh/SharedData/zhanh/202601_midisemi_data}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_BASE}/score_hpt/workspaces}"
MAESTRO_DIR="${MAESTRO_DIR:-/media/mengh/SharedData/zhanh/Dataset/maestro-v3.0.0}"
DDSP_CKPT_ROOT="${DDSP_CKPT_ROOT:-${WORKSPACE_BASE}/ddsp-piano-pytorch}"

SEGMENT_LIST="${SEGMENT_LIST:-2 5 10}"
LOSS_TYPES="${LOSS_TYPES:-piano_ssm_spectral piano_ssm_spectral_plus_log_rms piano_ssm_spectral_plus_ddsp_loudness piano_ssm_combined_rm}"
DDSP_CKPTS="${DDSP_CKPTS:-}"
CKPT_EPOCH="${CKPT_EPOCH:-7}"

BATCH_SIZE="${BATCH_SIZE:-4}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
PROXY_WEIGHT="${PROXY_WEIGHT:-1.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

resolve_ddsp_ckpt() {
  local segment="$1"
  local seg_tag="${segment%.0}"
  local path="${DDSP_CKPT_ROOT}/workspaces_unified_${seg_tag}s/models/ckpts/ddsp-piano_epoch_${CKPT_EPOCH}_params.pt"
  [ -f "${path}" ] && printf '%s\n' "${path}"
}

has_hdf5() {
  find "$1" -type f -name '*.h5' -print -quit 2>/dev/null | grep -q .
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

if [ ! -f "${MAESTRO_DIR}/maestro-v3.0.0.csv" ]; then
  echo "MAESTRO metadata file not found under: ${MAESTRO_DIR}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} not found in PATH. Please activate the correct environment first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

HDF5_DIR="${WORKSPACE_DIR}/hdf5s/maestro_sr22050"
if [ ! -d "${HDF5_DIR}" ] || ! has_hdf5 "${HDF5_DIR}"; then
  echo "Packed MAESTRO HDF5 dataset not found. Running preprocessing first."
  "${PYTHON_BIN}" pytorch/data_generator.py pack_maestro_dataset_to_hdf5 \
    "exp.workspace=${WORKSPACE_DIR}" \
    "feature.sample_rate=22050" \
    "dataset.maestro_dir=${MAESTRO_DIR}"
fi

run_one() {
  local segment="$1"
  local ckpt="$2"
  local audio_loss="$3"
  local seg_tag="${segment%.0}"

  echo "============================================================"
  echo "Route III ablation"
  echo "Dataset           : maestro"
  echo "Proxy checkpoint  : ${ckpt}"
  echo "Backend seg (s)   : ${segment}"
  echo "Audio loss        : ${audio_loss}"
  echo "============================================================"

  "${PYTHON_BIN}" pytorch/train_ddsp.py \
    "exp.workspace=${WORKSPACE_DIR}" \
    "exp.batch_size=${BATCH_SIZE}" \
    "dataset.train_set=maestro" \
    "dataset.test_set=maestro" \
    "dataset.eval_sets=[train,maestro]" \
    "model.type=hpt" \
    "model.input2=onset" \
    "model.input3=frame" \
    "score_informed.method=note_editor" \
    "loss.supervised_weight=${SUPERVISED_WEIGHT}" \
    "loss.proxy_weight=${PROXY_WEIGHT}" \
    "loss.velocity_prior_weight=${PRIOR_WEIGHT}" \
    "proxy.enabled=true" \
    "proxy.type=diffsynth_piano" \
    "proxy.project_root=${DDSP_PROJECT_ROOT}" \
    "proxy.checkpoint=${ckpt}" \
    "proxy.backend_segment_seconds=${segment}" \
    "proxy.audio_loss.type=${audio_loss}" \
    "wandb.comment=route3_maestro_ddsp_${seg_tag}s_${audio_loss}" \
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
    echo "Skip ${segment}s: missing DDSP-Piano ckpt under ${DDSP_CKPT_ROOT}" >&2
    continue
  fi
  for audio_loss in ${LOSS_TYPES}; do
    run_one "${segment}" "${ckpt}" "${audio_loss}"
  done
done
