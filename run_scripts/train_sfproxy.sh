#!/usr/bin/env bash
set -euo pipefail

# Manual config.
# Edit these defaults here, then run:
#   bash run_scripts/train_sfproxy.sh
#
# This script intentionally launches exactly one SFProxy run.
# It exports teacher data once and trains one proxy once.

DEFAULT_MACHINE="5090"              # 3090 or 5090
DEFAULT_GPU="0"                     # exported to CUDA_VISIBLE_DEVICES
DEFAULT_INSTRUMENT="piano"          # piano or guitar
DEFAULT_SEGMENT_SECONDS="2"         # usually 2 / 5 / 10 for piano, 2 / 5 for guitar
DEFAULT_SAMPLER_PRESET="mixed_v2"   # coverage_shared_legacy | realism_shared_legacy | coverage_v2 | realism_v2 | mixed_v2
DEFAULT_BOUNDARY_MODE="discovered"  # default | fixed | discovered
DEFAULT_SOUNDFONT=""                # absolute path, relative path, or basename under SOUNDFONT_DIR
DEFAULT_DISCOVER_BOUNDARIES="0"     # set 1 to auto-create boundary json when missing
DEFAULT_EXPORT_AUDIO="0"            # 0 disable, -1 all, N first N exports
DEFAULT_RESET_OUTPUTS="1"           # 1 -> clear old export/train dirs for this run tag

DEFAULT_PROJECT_NAME="synth-proxy_v1"
DEFAULT_TRAIN_SIZE="20000"
DEFAULT_VAL_SIZE="2000"
DEFAULT_EPOCHS="200"
DEFAULT_BATCH_SIZE="64"
DEFAULT_VAL_BATCH_SIZE="64"
DEFAULT_NUM_WORKERS="4"
DEFAULT_VAL_NUM_WORKERS="2"

DEFAULT_WANDB_PROJECT="sfproxy"
DEFAULT_WANDB_GROUP_PREFIX="sfproxy_manual"

DEFAULT_EXTRA_EXPORT_OVERRIDES=""
DEFAULT_EXTRA_TRAIN_OVERRIDES=""

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${ROOT_DIR}/synth-proxy"
PYTHON_BIN="${PYTHON_BIN:-python}"

MACHINE="${MACHINE:-${DEFAULT_MACHINE}}"
case "${MACHINE}" in
  3090)
    DEFAULT_WORKSPACE_BASE="${ROOT_DIR}/_runs"
    DEFAULT_ANALYSIS_DIR="${ROOT_DIR}/data_analysis"
    DEFAULT_SOUNDFONT_DIR="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfont"
    DEFAULT_PIANO_INSTRUMENT_PATH="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfront/SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2"
    DEFAULT_GUITAR_INSTRUMENT_PATH="${ROOT_DIR}/data/soundfonts/guitar.sf2"
    ;;
  5090)
    DEFAULT_WORKSPACE_BASE="/path/to/202601_midisemi_data"
    DEFAULT_ANALYSIS_DIR="${ROOT_DIR}/data_analysis"
    DEFAULT_SOUNDFONT_DIR="/path/to/soundfont"
    DEFAULT_PIANO_INSTRUMENT_PATH="/path/to/SalamanderGrandPiano-V3+20200602.sf2"
    DEFAULT_GUITAR_INSTRUMENT_PATH="/path/to/guitar.sf2"
    ;;
  *)
    echo "Unsupported MACHINE='${MACHINE}'. Expected '3090' or '5090'." >&2
    exit 1
    ;;
esac

GPU="${GPU:-${DEFAULT_GPU}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"

WORKSPACE_BASE="${WORKSPACE_BASE:-${DEFAULT_WORKSPACE_BASE}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${DEFAULT_ANALYSIS_DIR}}"
PROJECT_NAME="${PROJECT_NAME:-${DEFAULT_PROJECT_NAME}}"
SOUNDFONT_DIR="${SOUNDFONT_DIR:-${DEFAULT_SOUNDFONT_DIR}}"

# Fixed export / train contract for this repo.
TRAIN_SEED_OFFSET="0"
VAL_SEED_OFFSET="1000"
ACCELERATOR="gpu"
DEVICES="1"
FEATURE_HOP="221"

INSTRUMENT="${INSTRUMENT:-${DEFAULT_INSTRUMENT}}"
case "${INSTRUMENT}" in
  piano)
    EXPORT_CONFIG="data_piano"
    TRAIN_CONFIG="train"
    DATASET_NAME="piano"
    INSTRUMENT_NAME="salamander_piano"
    DEFAULT_INSTRUMENT_PATH="${DEFAULT_PIANO_INSTRUMENT_PATH}"
    DEFAULT_BOUNDARY_JSON="${ANALYSIS_DIR}/stats/sfproxy_boundaries/salamander_piano_boundaries.json"
    PITCH_MIN=21
    PITCH_MAX=108
    PITCH_STEP=6
    REGISTER_SPLITS=(48 72)
    ;;
  guitar)
    EXPORT_CONFIG="data_guitar"
    TRAIN_CONFIG="train"
    DATASET_NAME="guitar"
    INSTRUMENT_NAME="guitar"
    DEFAULT_INSTRUMENT_PATH="${DEFAULT_GUITAR_INSTRUMENT_PATH}"
    DEFAULT_BOUNDARY_JSON="${ANALYSIS_DIR}/stats/sfproxy_boundaries/guitar_boundaries.json"
    PITCH_MIN=42
    PITCH_MAX=72
    PITCH_STEP=3
    REGISTER_SPLITS=(52 64)
    ;;
  *)
    echo "Unsupported INSTRUMENT='${INSTRUMENT}'. Expected 'piano' or 'guitar'." >&2
    exit 1
    ;;
esac

SEGMENT_SECONDS="${SEGMENT_SECONDS:-${DEFAULT_SEGMENT_SECONDS}}"
SEGMENT_TAG="${SEGMENT_SECONDS%.0}"
SAMPLER_PRESET="${SAMPLER_PRESET:-${DEFAULT_SAMPLER_PRESET}}"
BOUNDARY_MODE="${BOUNDARY_MODE:-${DEFAULT_BOUNDARY_MODE}}"
SOUNDFONT="${SOUNDFONT:-${DEFAULT_SOUNDFONT}}"
DISCOVER_BOUNDARIES="${DISCOVER_BOUNDARIES:-${DEFAULT_DISCOVER_BOUNDARIES}}"
EXPORT_AUDIO="${EXPORT_AUDIO:-${DEFAULT_EXPORT_AUDIO}}"
RESET_OUTPUTS="${RESET_OUTPUTS:-${DEFAULT_RESET_OUTPUTS}}"

TRAIN_SIZE="${TRAIN_SIZE:-${DEFAULT_TRAIN_SIZE}}"
VAL_SIZE="${VAL_SIZE:-${DEFAULT_VAL_SIZE}}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${DEFAULT_VAL_BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-${DEFAULT_VAL_NUM_WORKERS}}"
BOUNDARY_JSON="${BOUNDARY_JSON:-${DEFAULT_BOUNDARY_JSON}}"

TASK_TAG="${TASK_TAG:-${INSTRUMENT_NAME}_${SAMPLER_PRESET}_${SEGMENT_TAG}s_${BOUNDARY_MODE}}"
WANDB_PROJECT="${WANDB_PROJECT:-${DEFAULT_WANDB_PROJECT}}"
WANDB_GROUP="${WANDB_GROUP:-${DEFAULT_WANDB_GROUP_PREFIX}_${INSTRUMENT_NAME}}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-${SAMPLER_PRESET}_${BOUNDARY_MODE}_${SEGMENT_TAG}s}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TASK_TAG}}"

EXTRA_EXPORT_OVERRIDES="${EXTRA_EXPORT_OVERRIDES:-${DEFAULT_EXTRA_EXPORT_OVERRIDES}}"
EXTRA_TRAIN_OVERRIDES="${EXTRA_TRAIN_OVERRIDES:-${DEFAULT_EXTRA_TRAIN_OVERRIDES}}"

TEACHER_ROOT="${WORKSPACE_BASE}/${PROJECT_NAME}/teacher_data/${INSTRUMENT_NAME}/${TASK_TAG}"
TRAIN_EXPORT_DIR="${TEACHER_ROOT}/train"
VAL_EXPORT_DIR="${TEACHER_ROOT}/val"
TRAIN_RUN_DIR="${WORKSPACE_BASE}/${PROJECT_NAME}/proxy/logs/sfproxy/${TASK_TAG}"

resolve_instrument_path() {
  local raw="$1"
  local candidate=""

  if [[ -z "${raw}" ]]; then
    echo "${DEFAULT_INSTRUMENT_PATH}"
    return 0
  fi

  if [[ -f "${raw}" ]]; then
    echo "${raw}"
    return 0
  fi

  if [[ -f "${SOUNDFONT_DIR}/${raw}" ]]; then
    echo "${SOUNDFONT_DIR}/${raw}"
    return 0
  fi

  if [[ "${raw}" != *.* ]]; then
    if [[ -f "${SOUNDFONT_DIR}/${raw}.sf2" ]]; then
      echo "${SOUNDFONT_DIR}/${raw}.sf2"
      return 0
    fi
    if [[ -f "${SOUNDFONT_DIR}/${raw}.sfz" ]]; then
      echo "${SOUNDFONT_DIR}/${raw}.sfz"
      return 0
    fi
  fi

  candidate="${SOUNDFONT_DIR}/${raw}"
  echo "${candidate}"
  return 0
}

INSTRUMENT_PATH="$(resolve_instrument_path "${INSTRUMENT_PATH:-${SOUNDFONT}}")"
if [[ ! -f "${INSTRUMENT_PATH}" ]]; then
  echo "Instrument file not found: ${INSTRUMENT_PATH}" >&2
  echo "You can set SOUNDFONT to an absolute path, or to a file/basename under: ${SOUNDFONT_DIR}" >&2
  exit 1
fi

case "${INSTRUMENT_PATH##*.}" in
  sf2|SF2)
    ;;
  sfz|SFZ)
    ;;
  *)
    echo "Unsupported instrument format: ${INSTRUMENT_PATH}" >&2
    echo "Expected an .sf2 or .sfz file for the current SFProxy pipeline." >&2
    exit 1
    ;;
esac

if [[ "${SAMPLER_PRESET}" == *legacy* && "${BOUNDARY_MODE}" == "discovered" ]]; then
  echo "Legacy samplers share one velocity inside each chord, so they cannot use discovered boundaries." >&2
  echo "Use BOUNDARY_MODE=default or fixed, or switch to a v2 sampler preset." >&2
  exit 1
fi

mkdir -p "${WORKSPACE_BASE}" "${ANALYSIS_DIR}/stats/sfproxy_boundaries"

cd "${ROOT_DIR}"

run_boundary_discovery() {
  if [[ -f "${BOUNDARY_JSON}" ]]; then
    echo "Using existing boundary JSON: ${BOUNDARY_JSON}"
    return 0
  fi
  if [[ "${DISCOVER_BOUNDARIES}" != "1" ]]; then
    echo "Boundary JSON missing (${BOUNDARY_JSON}); sampler will fall back to default [0.33, 0.66]." >&2
    return 0
  fi

  echo "Discovering velocity boundaries from the source instrument -> ${BOUNDARY_JSON}"
  "${PYTHON_BIN}" "${ROOT_DIR}/synth-proxy/src/sfproxy/tools/discover_velocity_boundaries.py" \
    --instrument_path "${INSTRUMENT_PATH}" \
    --instrument_name "${INSTRUMENT_NAME}" \
    --bank 0 \
    --program 0 \
    --sr 22050 \
    --seg_len_s "${SEGMENT_SECONDS}" \
    --pitch_min "${PITCH_MIN}" \
    --pitch_max "${PITCH_MAX}" \
    --pitch_step "${PITCH_STEP}" \
    --register_splits "${REGISTER_SPLITS[@]}" \
    --hop "${FEATURE_HOP}" \
    --out_json "${BOUNDARY_JSON}"
}

make_boundary_overrides() {
  local mode="$1"
  local -n out_ref=$2
  out_ref=()
  case "${mode}" in
    default)
      ;;
    fixed)
      out_ref+=(
        "sampler_options.coverage_v2.velocity_boundary_path=''"
        "sampler_options.coverage_v2.velocity_boundary_strategy=global"
        "sampler_options.realism_v2.velocity_boundary_path=''"
        "sampler_options.realism_v2.velocity_boundary_strategy=global"
        "sampler_options.mixed_v2.components.boundary.velocity_boundary_path=''"
        "sampler_options.mixed_v2.components.boundary.velocity_boundary_strategy=global"
        "sampler_options.mixed_v2.components.coverage.velocity_boundary_path=''"
        "sampler_options.mixed_v2.components.coverage.velocity_boundary_strategy=global"
        "sampler_options.mixed_v2.components.realism.velocity_boundary_path=''"
        "sampler_options.mixed_v2.components.realism.velocity_boundary_strategy=global"
        "sampler_options.mixed_v2.components.stress.velocity_boundary_path=''"
        "sampler_options.mixed_v2.components.stress.velocity_boundary_strategy=global"
      )
      ;;
    discovered)
      run_boundary_discovery
      out_ref+=(
        "sampler_options.coverage_v2.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.realism_v2.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.mixed_v2.components.boundary.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.mixed_v2.components.coverage.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.mixed_v2.components.realism.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.mixed_v2.components.stress.velocity_boundary_path=${BOUNDARY_JSON}"
      )
      ;;
    *)
      echo "Unsupported BOUNDARY_MODE='${mode}'. Expected default, fixed, or discovered." >&2
      exit 1
      ;;
  esac
}

boundary_overrides=()
make_boundary_overrides "${BOUNDARY_MODE}" boundary_overrides

read -r -a extra_export_args <<< "${EXTRA_EXPORT_OVERRIDES}"
read -r -a extra_train_args <<< "${EXTRA_TRAIN_OVERRIDES}"

echo "============================================================"
echo "SFProxy single-run config"
echo "Machine         : ${MACHINE}"
echo "CUDA devices    : ${CUDA_VISIBLE_DEVICES}"
echo "Instrument      : ${INSTRUMENT_NAME}"
echo "Instrument dir  : ${SOUNDFONT_DIR}"
echo "Source file     : ${INSTRUMENT_PATH}"
echo "Segment seconds : ${SEGMENT_SECONDS}"
echo "Sampler preset  : ${SAMPLER_PRESET}"
echo "Boundary mode   : ${BOUNDARY_MODE}"
echo "Boundary JSON   : ${BOUNDARY_JSON}"
echo "Train size      : ${TRAIN_SIZE}"
echo "Val size        : ${VAL_SIZE}"
echo "Task tag        : ${TASK_TAG}"
echo "Teacher root    : ${TEACHER_ROOT}"
echo "Train run dir   : ${TRAIN_RUN_DIR}"
echo "W&B project     : ${WANDB_PROJECT}"
echo "W&B group       : ${WANDB_GROUP}"
echo "W&B run name    : ${WANDB_RUN_NAME}"
echo "============================================================"

if [[ "${RESET_OUTPUTS}" == "1" ]]; then
  rm -rf "${TRAIN_EXPORT_DIR}" "${VAL_EXPORT_DIR}" "${TRAIN_RUN_DIR}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/synth-proxy/src/sfproxy/export_dataset_pkl.py" \
  --config-name "${EXPORT_CONFIG}" \
  "paths.repo_root=${ROOT_DIR}" \
  "paths.workspace_dir=${WORKSPACE_BASE}" \
  "paths.project_name=${PROJECT_NAME}" \
  "paths.analysis_dir=${ANALYSIS_DIR}" \
  "instrument.path=${INSTRUMENT_PATH}" \
  "instrument.seg_len_s=${SEGMENT_SECONDS}" \
  "sampler_preset=${SAMPLER_PRESET}" \
  "dataset_size=${TRAIN_SIZE}" \
  "start_index=0" \
  "end_index=${TRAIN_SIZE}" \
  "seed_offset=${TRAIN_SEED_OFFSET}" \
  "feature.hop=${FEATURE_HOP}" \
  "export_audio=${EXPORT_AUDIO}" \
  "hydra.run.dir=${TRAIN_EXPORT_DIR}" \
  "${boundary_overrides[@]}" \
  "${extra_export_args[@]}"

"${PYTHON_BIN}" "${ROOT_DIR}/synth-proxy/src/sfproxy/export_dataset_pkl.py" \
  --config-name "${EXPORT_CONFIG}" \
  "paths.repo_root=${ROOT_DIR}" \
  "paths.workspace_dir=${WORKSPACE_BASE}" \
  "paths.project_name=${PROJECT_NAME}" \
  "paths.analysis_dir=${ANALYSIS_DIR}" \
  "instrument.path=${INSTRUMENT_PATH}" \
  "instrument.seg_len_s=${SEGMENT_SECONDS}" \
  "sampler_preset=${SAMPLER_PRESET}" \
  "dataset_size=${VAL_SIZE}" \
  "start_index=0" \
  "end_index=${VAL_SIZE}" \
  "seed_offset=${VAL_SEED_OFFSET}" \
  "feature.hop=${FEATURE_HOP}" \
  "export_audio=0" \
  "hydra.run.dir=${VAL_EXPORT_DIR}" \
  "${boundary_overrides[@]}" \
  "${extra_export_args[@]}"

"${PYTHON_BIN}" "${ROOT_DIR}/synth-proxy/src/sfproxy/train.py" \
  --config-name "${TRAIN_CONFIG}" \
  "task_name=${TASK_TAG}" \
  "paths.repo_root=${ROOT_DIR}" \
  "paths.workspace_dir=${WORKSPACE_BASE}" \
  "paths.project_name=${PROJECT_NAME}" \
  "dataset.train.path=${TRAIN_EXPORT_DIR}" \
  "dataset.val.path=${VAL_EXPORT_DIR}" \
  "dataset.name=${DATASET_NAME}" \
  "dataset.train.instrument_name=${INSTRUMENT_NAME}" \
  "dataset.val.instrument_name=${INSTRUMENT_NAME}" \
  "dataset.train.loader.batch_size=${BATCH_SIZE}" \
  "dataset.train.loader.num_workers=${NUM_WORKERS}" \
  "dataset.val.loader.batch_size=${VAL_BATCH_SIZE}" \
  "dataset.val.loader.num_workers=${VAL_NUM_WORKERS}" \
  "trainer.max_epochs=${EPOCHS}" \
  "trainer.accelerator=${ACCELERATOR}" \
  "trainer.devices=${DEVICES}" \
  "logger.wandb.offline=false" \
  "logger.wandb.project=${WANDB_PROJECT}" \
  "logger.wandb.name=${WANDB_RUN_NAME}" \
  "logger.wandb.group=${WANDB_GROUP}" \
  "logger.wandb.job_type=${WANDB_JOB_TYPE}" \
  "hydra.run.dir=${TRAIN_RUN_DIR}" \
  "${extra_train_args[@]}"
