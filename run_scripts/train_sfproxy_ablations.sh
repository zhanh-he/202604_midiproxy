#!/usr/bin/env bash
set -euo pipefail

# SFProxy ablations.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

INSTRUMENT="${1:-${INSTRUMENT:-piano}}"
DEFAULT_PIANO_INSTRUMENT_PATH="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfont/SalamanderGrandPiano-SFZ/SalamanderGrandPianoV3.sfz"
DEFAULT_GUITAR_INSTRUMENT_PATH="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfont/SpanishClassicalGuitar-SFZ/SpanishClassicalGuitar-20190618.sfz"

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
DISCOVER_BOUNDARIES="${DISCOVER_BOUNDARIES:-0}"

EXTRA_EXPORT_OVERRIDES="${EXTRA_EXPORT_OVERRIDES:-}"
EXTRA_TRAIN_OVERRIDES="${EXTRA_TRAIN_OVERRIDES:-}"

TRAIN_CONFIG="train"
DATASET_NAME="${INSTRUMENT}"

case "${INSTRUMENT}" in
  piano)
    EXPORT_CONFIG="data_piano"
    INSTRUMENT_NAME="salamander_piano"
    DEFAULT_SEGMENT_LIST="2 5 10"
    DEFAULT_SAMPLER_PRESETS="coverage_shared_legacy coverage_v2 mixed_v2"
    DEFAULT_BOUNDARY_MODE_LIST="default fixed discovered"
    DEFAULT_INSTRUMENT_PATH="${DEFAULT_PIANO_INSTRUMENT_PATH}"
    DEFAULT_BOUNDARY_JSON="${ANALYSIS_DIR}/stats/sfproxy_boundaries/salamander_piano_boundaries.json"
    PITCH_MIN=21
    PITCH_MAX=108
    PITCH_STEP=6
    REGISTER_SPLITS=(48 72)
    ;;
  guitar)
    EXPORT_CONFIG="data_guitar"
    INSTRUMENT_NAME="guitar"
    DEFAULT_SEGMENT_LIST="2 5"
    DEFAULT_SAMPLER_PRESETS="coverage_shared_legacy coverage_v2 mixed_v2"
    DEFAULT_BOUNDARY_MODE_LIST="default fixed discovered"
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

INSTRUMENT_PATH="${INSTRUMENT_PATH:-${DEFAULT_INSTRUMENT_PATH}}"
SEGMENT_LIST="${SEGMENT_LIST:-${DEFAULT_SEGMENT_LIST}}"
SAMPLER_PRESETS="${SAMPLER_PRESETS:-${DEFAULT_SAMPLER_PRESETS}}"
BOUNDARY_MODE_LIST="${BOUNDARY_MODE_LIST:-${DEFAULT_BOUNDARY_MODE_LIST}}"
BOUNDARY_JSON="${BOUNDARY_JSON:-${DEFAULT_BOUNDARY_JSON}}"

mkdir -p "${WORKSPACE_BASE}" "${ANALYSIS_DIR}/stats/sfproxy_boundaries"

cd "${ROOT_DIR}"

run_boundary_discovery() {
  local segment_seconds="$1"

  if [[ -f "${BOUNDARY_JSON}" ]]; then
    echo "Using existing boundary JSON: ${BOUNDARY_JSON}"
    return 0
  fi

  if [[ "${DISCOVER_BOUNDARIES}" != "1" ]]; then
    echo "Boundary JSON missing (${BOUNDARY_JSON}); fallback to default [0.33, 0.66]." >&2
    return 0
  fi

  echo "Discovering velocity boundaries -> ${BOUNDARY_JSON}"
  python "${ROOT_DIR}/synth-proxy/src/sfproxy/tools/discover_velocity_boundaries.py" \
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
  local segment_seconds="$2"
  local -n out_ref=$3
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
      run_boundary_discovery "${segment_seconds}"
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

read -r -a extra_export_args <<< "${EXTRA_EXPORT_OVERRIDES}"
read -r -a extra_train_args <<< "${EXTRA_TRAIN_OVERRIDES}"

for SEGMENT_SECONDS in ${SEGMENT_LIST}; do
  SEGMENT_TAG="${SEGMENT_SECONDS%.0}"

  for SAMPLER_PRESET in ${SAMPLER_PRESETS}; do
    for BOUNDARY_MODE in ${BOUNDARY_MODE_LIST}; do
      if [[ "${SAMPLER_PRESET}" == *legacy* && "${BOUNDARY_MODE}" == "discovered" ]]; then
        continue
      fi

      echo "============================================================"
      echo "Instrument      : ${INSTRUMENT_NAME}"
      echo "Source file     : ${INSTRUMENT_PATH}"
      echo "Segment seconds : ${SEGMENT_SECONDS}"
      echo "Sampler preset  : ${SAMPLER_PRESET}"
      echo "Boundary mode   : ${BOUNDARY_MODE}"
      echo "============================================================"

      boundary_overrides=()
      make_boundary_overrides "${BOUNDARY_MODE}" "${SEGMENT_SECONDS}" boundary_overrides

      python "${ROOT_DIR}/synth-proxy/src/sfproxy/export_dataset_pkl.py" \
        --config-name "${EXPORT_CONFIG}" \
        "paths.repo_root=${ROOT_DIR}" \
        "paths.workspace_dir=${WORKSPACE_BASE}" \
        "paths.analysis_dir=${ANALYSIS_DIR}" \
        "instrument.path=${INSTRUMENT_PATH}" \
        "instrument.seg_len_s=${SEGMENT_SECONDS}" \
        "sampler_preset=${SAMPLER_PRESET}" \
        "boundary_mode=${BOUNDARY_MODE}" \
        "split=train" \
        "reset_output_dir=true" \
        "${boundary_overrides[@]}" \
        "${extra_export_args[@]}"

      python "${ROOT_DIR}/synth-proxy/src/sfproxy/export_dataset_pkl.py" \
        --config-name "${EXPORT_CONFIG}" \
        "paths.repo_root=${ROOT_DIR}" \
        "paths.workspace_dir=${WORKSPACE_BASE}" \
        "paths.analysis_dir=${ANALYSIS_DIR}" \
        "instrument.path=${INSTRUMENT_PATH}" \
        "instrument.seg_len_s=${SEGMENT_SECONDS}" \
        "sampler_preset=${SAMPLER_PRESET}" \
        "boundary_mode=${BOUNDARY_MODE}" \
        "split=val" \
        "reset_output_dir=true" \
        "${boundary_overrides[@]}" \
        "${extra_export_args[@]}"

      python "${ROOT_DIR}/synth-proxy/src/sfproxy/train.py" \
        --config-name "${TRAIN_CONFIG}" \
        "paths.repo_root=${ROOT_DIR}" \
        "paths.workspace_dir=${WORKSPACE_BASE}" \
        "sampler_preset=${SAMPLER_PRESET}" \
        "segment_seconds=${SEGMENT_SECONDS}" \
        "boundary_mode=${BOUNDARY_MODE}" \
        "dataset.name=${DATASET_NAME}" \
        "dataset.train.instrument_name=${INSTRUMENT_NAME}" \
        "dataset.val.instrument_name=${INSTRUMENT_NAME}" \
        "reset_output_dir=true" \
        "${extra_train_args[@]}"
    done
  done
done
