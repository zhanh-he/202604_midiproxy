#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROJECT_DIR="${ROOT_DIR}/score_hpt"
PROJECT_ROOT="${PROJECT_ROOT:-${ROOT_DIR}/synth-proxy}"

WORKSPACE_BASE="${WORKSPACE_BASE:-/media/mengh/SharedData/zhanh/202601_midisemi_data}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_BASE}/score_hpt/workspaces}"
MAESTRO_DIR="${MAESTRO_DIR:-/media/mengh/SharedData/zhanh/Dataset/maestro-v3.0.0}"
INSTRUMENT_NAME="${INSTRUMENT_NAME:-salamander_piano}"
BOUNDARY_MODE="${BOUNDARY_MODE:-default}"
SEGMENT_LIST="${SEGMENT_LIST:-2 5 10}"
SAMPLERS="${SAMPLERS:-coverage mixed realism}"
LOSS_TYPES="${LOSS_TYPES:-smooth_l1 l1 mse}"
SFPROXY_CKPT_ROOT="${SFPROXY_CKPT_ROOT:-${WORKSPACE_BASE}/synth-proxy/proxy/checkpoints/${INSTRUMENT_NAME}}"
PROXY_CKPT="${PROXY_CKPT:-${1:-}}"

BATCH_SIZE="${BATCH_SIZE:-4}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
PROXY_WEIGHT="${PROXY_WEIGHT:-1.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.0}"
WARMUP_ITERS="${WARMUP_ITERS:-0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

sampler_preset() {
  case "$1" in
    coverage|coverage_v2) echo "coverage_v2" ;;
    mixed|mixed_v2) echo "mixed_v2" ;;
    realism|realism_v2) echo "realism_v2" ;;
    *) return 1 ;;
  esac
}

has_hdf5() {
  find "$1" -type f -name '*.h5' -print -quit 2>/dev/null | grep -q .
}

resolve_sfproxy_ckpt() {
  local sampler="$1"
  local segment="$2"
  local seg_tag="${segment%.0}s"
  local preset
  local dir
  local latest
  preset="$(sampler_preset "${sampler}")" || return 0
  for dir in "${SFPROXY_CKPT_ROOT}"/piano_"${INSTRUMENT_NAME}"_"${preset}"*_"${seg_tag}"_"${BOUNDARY_MODE}"; do
    [ -d "${dir}" ] || continue
    latest="$(find "${dir}" -maxdepth 1 -type f -name '*last*.ckpt' | sort -V | tail -n 1)"
    [ -n "${latest}" ] && printf '%s\n' "${latest}" && return 0
    latest="$(find "${dir}" -maxdepth 1 -type f -name '*loss*.ckpt' | sort -V | tail -n 1)"
    [ -n "${latest}" ] && printf '%s\n' "${latest}" && return 0
    latest="$(find "${dir}" -maxdepth 1 -type f -name '*.ckpt' | sort -V | tail -n 1)"
    [ -n "${latest}" ] && printf '%s\n' "${latest}" && return 0
  done
}

mkdir -p "${WORKSPACE_DIR}"
read -r -a extra_args <<< "${EXTRA_OVERRIDES}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "SFProxy project root not found: ${PROJECT_ROOT}" >&2
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
  local sampler="$2"
  local ckpt="$3"
  local loss="$4"
  local seg_tag="${segment%.0}"

  echo "============================================================"
  echo "Route IV ablation"
  echo "Dataset           : maestro"
  echo "Proxy checkpoint  : ${ckpt}"
  echo "Backend seg (s)   : ${segment}"
  echo "Sampler           : ${sampler}"
  echo "Proxy loss        : ${loss}"
  echo "============================================================"

  "${PYTHON_BIN}" pytorch/train_proxy.py \
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
    "proxy.type=diffproxy" \
    "proxy.project_root=${PROJECT_ROOT}" \
    "proxy.checkpoint=${ckpt}" \
    "proxy.backend_segment_seconds=${segment}" \
    "proxy.warmup_iterations=${WARMUP_ITERS}" \
    "proxy.supervision.hop_size=221" \
    "proxy.sfproxy.instrument_name=${INSTRUMENT_NAME}" \
    "proxy.sfproxy.loss_type=${loss}" \
    "proxy.sfproxy.use_gt_aligned_note_events=true" \
    "proxy.sfproxy.feature.hop=221" \
    "wandb.comment=route4_maestro_${sampler}_${seg_tag}s_${loss}" \
    "${extra_args[@]}"
}

if [ -n "${PROXY_CKPT}" ]; then
  if [ ! -f "${PROXY_CKPT}" ]; then
    echo "SFProxy checkpoint not found: ${PROXY_CKPT}" >&2
    exit 1
  fi
  for segment in ${SEGMENT_LIST}; do
    for loss in ${LOSS_TYPES}; do
      run_one "${segment}" "manual" "${PROXY_CKPT}" "${loss}"
    done
  done
  exit 0
fi

for segment in ${SEGMENT_LIST}; do
  for sampler in ${SAMPLERS}; do
    ckpt="$(resolve_sfproxy_ckpt "${sampler}" "${segment}")"
    if [ -z "${ckpt}" ]; then
      echo "Skip ${sampler} ${segment}s: missing SFProxy ckpt under ${SFPROXY_CKPT_ROOT}" >&2
      continue
    fi
    for loss in ${LOSS_TYPES}; do
      run_one "${segment}" "${sampler}" "${ckpt}" "${loss}"
    done
  done
done
