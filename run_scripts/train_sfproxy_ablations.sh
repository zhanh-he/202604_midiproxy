#!/usr/bin/env bash
set -euo pipefail

# SFProxy teacher-data / proxy-training ablations.
# Examples:
#   bash run_scripts/train_sfproxy_ablations.sh piano
#   SEGMENT_LIST="2 5 10" SAMPLER_PRESETS="coverage_shared_legacy mixed_v2" bash run_scripts/train_sfproxy_ablations.sh piano
#   INSTRUMENT=guitar BOUNDARY_MODE_LIST="fixed discovered" INSTRUMENT_PATH=/path/to/guitar.sf2 bash run_scripts/train_sfproxy_ablations.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INSTRUMENT="${1:-${INSTRUMENT:-piano}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROJECT_ROOT="${PROJECT_ROOT:-${ROOT_DIR}/synth-proxy}"
WORKSPACE_BASE="${WORKSPACE_BASE:-${ROOT_DIR}/_runs}"
PROJECT_NAME="${PROJECT_NAME:-synth-proxy_v1}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${ROOT_DIR}/data_analysis}"
TRAIN_SIZE="${TRAIN_SIZE:-20000}"
VAL_SIZE="${VAL_SIZE:-2000}"
TRAIN_SEED_OFFSET="${TRAIN_SEED_OFFSET:-0}"
VAL_SEED_OFFSET="${VAL_SEED_OFFSET:-1000}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
DISCOVER_BOUNDARIES="${DISCOVER_BOUNDARIES:-0}"
EXPORT_AUDIO="${EXPORT_AUDIO:-0}"
EXTRA_EXPORT_OVERRIDES="${EXTRA_EXPORT_OVERRIDES:-}"
EXTRA_TRAIN_OVERRIDES="${EXTRA_TRAIN_OVERRIDES:-}"

case "${INSTRUMENT}" in
  piano)
    EXPORT_CONFIG="data_piano"
    TRAIN_CONFIG="train"
    DATASET_NAME="piano"
    INSTRUMENT_NAME="salamander_piano"
    DEFAULT_SEGMENT_LIST="2 5 10"
    DEFAULT_SAMPLER_PRESETS="coverage_shared_legacy coverage_v2 mixed_v2"
    DEFAULT_BOUNDARY_MODE_LIST="default fixed discovered"
    DEFAULT_INSTRUMENT_PATH="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfront/SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2"
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
    DEFAULT_SEGMENT_LIST="2 5"
    DEFAULT_SAMPLER_PRESETS="coverage_shared_legacy coverage_v2 mixed_v2"
    DEFAULT_BOUNDARY_MODE_LIST="default fixed discovered"
    DEFAULT_INSTRUMENT_PATH="${ROOT_DIR}/data/soundfonts/guitar.sf2"
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

INSTRUMENT_PATH="${INSTRUMENT_PATH:-${DEFAULT_INSTRUMENT_PATH}}"
SEGMENT_LIST="${SEGMENT_LIST:-${DEFAULT_SEGMENT_LIST}}"
SAMPLER_PRESETS="${SAMPLER_PRESETS:-${DEFAULT_SAMPLER_PRESETS}}"
BOUNDARY_MODE_LIST="${BOUNDARY_MODE_LIST:-${DEFAULT_BOUNDARY_MODE_LIST}}"
BOUNDARY_JSON="${BOUNDARY_JSON:-${DEFAULT_BOUNDARY_JSON}}"

if [[ ! -f "${INSTRUMENT_PATH}" ]]; then
  echo "Instrument file not found: ${INSTRUMENT_PATH}" >&2
  echo "Set INSTRUMENT_PATH=/absolute/path/to/your.sf2 or your.sfz before running this script." >&2
  exit 1
fi

case "${INSTRUMENT_PATH##*.}" in
  sf2|SF2|sfz|SFZ)
    ;;
  *)
    echo "Unsupported instrument format: ${INSTRUMENT_PATH}" >&2
    echo "Expected an .sf2 or .sfz file." >&2
    exit 1
    ;;
esac

mkdir -p "${WORKSPACE_BASE}" "${ANALYSIS_DIR}/stats/sfproxy_boundaries"

cd "${ROOT_DIR}"

run_boundary_discovery() {
  local segment_seconds="$1"
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
    --seg_len_s "${segment_seconds}" \
    --pitch_min "${PITCH_MIN}" \
    --pitch_max "${PITCH_MAX}" \
    --pitch_step "${PITCH_STEP}" \
    --register_splits "${REGISTER_SPLITS[@]}" \
    --hop 221 \
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
      run_boundary_discovery "${CURRENT_SEGMENT_SECONDS}"
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
      echo "Unsupported boundary mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

for CURRENT_SEGMENT_SECONDS in ${SEGMENT_LIST}; do
  SEGMENT_TAG="${CURRENT_SEGMENT_SECONDS%.0}"
  for SAMPLER_PRESET in ${SAMPLER_PRESETS}; do
    for BOUNDARY_MODE in ${BOUNDARY_MODE_LIST}; do
      if [[ "${SAMPLER_PRESET}" == *legacy* && "${BOUNDARY_MODE}" == "discovered" ]]; then
        # Legacy samplers share one velocity inside each chord, so discovered boundaries do not apply.
        continue
      fi

      BOUNDARY_TAG="${BOUNDARY_MODE}"
      TASK_TAG="${INSTRUMENT_NAME}_${SAMPLER_PRESET}_${SEGMENT_TAG}s_${BOUNDARY_TAG}"
      WANDB_GROUP="sfproxy_ablations_${INSTRUMENT_NAME}"
      WANDB_JOB_TYPE="${SAMPLER_PRESET}_${BOUNDARY_MODE}_${SEGMENT_TAG}s"
      TEACHER_ROOT="${WORKSPACE_BASE}/${PROJECT_NAME}/teacher_data/${INSTRUMENT_NAME}/${TASK_TAG}"
      TRAIN_EXPORT_DIR="${TEACHER_ROOT}/train"
      VAL_EXPORT_DIR="${TEACHER_ROOT}/val"
      TRAIN_RUN_DIR="${WORKSPACE_BASE}/${PROJECT_NAME}/proxy/logs/sfproxy/${TASK_TAG}"

      echo "============================================================"
      echo "Task            : ${TASK_TAG}"
      echo "Source file     : ${INSTRUMENT_PATH}"
      echo "Segment seconds : ${CURRENT_SEGMENT_SECONDS}"
      echo "Sampler preset  : ${SAMPLER_PRESET}"
      echo "Boundary mode   : ${BOUNDARY_MODE}"
      echo "Teacher root    : ${TEACHER_ROOT}"
      echo "Train run dir   : ${TRAIN_RUN_DIR}"
      echo "============================================================"

      boundary_overrides=()
      make_boundary_overrides "${BOUNDARY_MODE}" boundary_overrides

      read -r -a extra_export_args <<< "${EXTRA_EXPORT_OVERRIDES}"
      read -r -a extra_train_args <<< "${EXTRA_TRAIN_OVERRIDES}"

      rm -rf "${TRAIN_EXPORT_DIR}" "${VAL_EXPORT_DIR}"

      "${PYTHON_BIN}" "${ROOT_DIR}/synth-proxy/src/sfproxy/export_dataset_pkl.py" \
        --config-name "${EXPORT_CONFIG}" \
        "paths.repo_root=${ROOT_DIR}" \
        "paths.workspace_dir=${WORKSPACE_BASE}" \
        "paths.project_name=${PROJECT_NAME}" \
        "paths.analysis_dir=${ANALYSIS_DIR}" \
        "instrument.path=${INSTRUMENT_PATH}" \
        "instrument.seg_len_s=${CURRENT_SEGMENT_SECONDS}" \
        "sampler_preset=${SAMPLER_PRESET}" \
        "dataset_size=${TRAIN_SIZE}" \
        "start_index=0" \
        "end_index=${TRAIN_SIZE}" \
        "seed_offset=${TRAIN_SEED_OFFSET}" \
        "feature.hop=221" \
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
        "instrument.seg_len_s=${CURRENT_SEGMENT_SECONDS}" \
        "sampler_preset=${SAMPLER_PRESET}" \
        "dataset_size=${VAL_SIZE}" \
        "start_index=0" \
        "end_index=${VAL_SIZE}" \
        "seed_offset=${VAL_SEED_OFFSET}" \
        "feature.hop=221" \
        "export_audio=0" \
        "hydra.run.dir=${VAL_EXPORT_DIR}" \
        "${boundary_overrides[@]}" \
        "${extra_export_args[@]}"

      rm -rf "${TRAIN_RUN_DIR}"
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
        "logger.wandb.name=${TASK_TAG}" \
        "logger.wandb.group=${WANDB_GROUP}" \
        "logger.wandb.job_type=${WANDB_JOB_TYPE}" \
        "hydra.run.dir=${TRAIN_RUN_DIR}" \
        "${extra_train_args[@]}"

      if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
        CKPT_PATH="${TRAIN_RUN_DIR}/checkpoints/last.ckpt"
        if [[ -f "${CKPT_PATH}" ]]; then
          echo "Diagnostics placeholder: add eval_monotonic / eval_velocity_recovery here if desired."
        else
          echo "Checkpoint not found for diagnostics: ${CKPT_PATH}" >&2
        fi
      fi
    done
  done
done
