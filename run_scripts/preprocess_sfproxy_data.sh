#!/usr/bin/env bash
set -euo pipefail

# Prepare reusable SFProxy v2 datasets.
#
# Plain `bash preprocess_sfproxy_data.sh` does the main thing:
# - instrument: piano
# - segments: 2 / 5 / 10 seconds
# - boundary mode: default
# - families: coverage_v2, realism_v2, mixed_v2
#
# mixed_v2 always means:
# - boundary + coverage + realism + stress
# - weights 0.30 / 0.40 / 0.20 / 0.10
#
# Clarifications:
# - TARGET_PRESET=all means "prepare coverage_v2, realism_v2, mixed_v2".
#   It is not a separate ablation axis.
# - BOUNDARY_MODE=discovered means "use auto-discovered velocity boundaries".
#   That can be used later as a boundary ablation, but it is not the default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Stable defaults for this repo / machine setup.
WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data"
ANALYSIS_DIR="${ROOT_DIR}/data_analysis"
PROJECT_NAME="synth-proxy_v1"
DATA_DIR="${WORKSPACE_BASE}/${PROJECT_NAME}/data"

DEFAULT_INSTRUMENT="piano"
DEFAULT_GUITAR_DATASET="francoisleduc"
DEFAULT_BOUNDARY_MODE="default"
DEFAULT_TARGET_PRESET="all"
DEFAULT_PIANO_SEGMENTS="2 5 10"
DEFAULT_GUITAR_SEGMENTS="2 5"
TRAIN_DATASET_SIZE=20000
VAL_DATASET_SIZE=2000

DEFAULT_MIXED_COMPONENT_NAMES="boundary coverage realism stress"
DEFAULT_MIXED_WEIGHTS="0.30 0.40 0.20 0.10"

DEFAULT_PIANO_INSTRUMENT_PATH="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz"
DEFAULT_GUITAR_INSTRUMENT_PATH="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfont/SpanishClassicalGuitar/SpanishClassicalGuitar-20190618.sfz"

# Only keep the few inputs that are actual experiment axes.
INSTRUMENT="${INSTRUMENT:-${DEFAULT_INSTRUMENT}}"
GUITAR_DATASET="${GUITAR_DATASET:-${DEFAULT_GUITAR_DATASET}}"
BOUNDARY_MODE="${BOUNDARY_MODE:-${DEFAULT_BOUNDARY_MODE}}"
TARGET_PRESET="${TARGET_PRESET:-${DEFAULT_TARGET_PRESET}}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-}"
SEGMENT_LIST="${SEGMENT_LIST:-}"
MIXED_COMPONENT_NAMES="${MIXED_COMPONENT_NAMES:-${DEFAULT_MIXED_COMPONENT_NAMES}}"
MIXED_WEIGHTS="${MIXED_WEIGHTS:-${DEFAULT_MIXED_WEIGHTS}}"

normalize_list() {
  local raw="$1"
  raw="${raw//,/ }"
  local -a items=()
  read -r -a items <<< "${raw}"
  echo "${items[*]}"
}

segment_tag() {
  local value="$1"
  if [[ "${value}" == *.* ]]; then
    value="${value%0}"
    value="${value%.}"
    value="${value//./p}"
  fi
  echo "${value}s"
}

component_to_preset() {
  case "$1" in
    boundary) echo "boundary_v2" ;;
    coverage) echo "coverage_v2" ;;
    realism) echo "realism_v2" ;;
    stress) echo "stress_v2" ;;
    *)
      echo "Unsupported mixed component '$1'." >&2
      exit 1
      ;;
  esac
}

component_short_name() {
  case "$1" in
    boundary) echo "b" ;;
    coverage) echo "c" ;;
    realism) echo "r" ;;
    stress) echo "s" ;;
    *)
      echo "$1"
      ;;
  esac
}

sanitize_weight_token() {
  local token="$1"
  token="${token//./p}"
  token="${token//-/_}"
  echo "${token}"
}

build_mixed_preset_name() {
  if [[ "${MIXED_COMPONENT_NAMES}" == "${DEFAULT_MIXED_COMPONENT_NAMES}" && "${MIXED_WEIGHTS}" == "${DEFAULT_MIXED_WEIGHTS}" ]]; then
    echo "mixed_v2"
    return 0
  fi

  local -a names=()
  local -a weights=()
  read -r -a names <<< "$(normalize_list "${MIXED_COMPONENT_NAMES}")"
  read -r -a weights <<< "$(normalize_list "${MIXED_WEIGHTS}")"

  local parts=()
  local idx
  for idx in "${!names[@]}"; do
    parts+=("$(component_short_name "${names[$idx]}")$(sanitize_weight_token "${weights[$idx]}")")
  done

  local IFS=_
  echo "mixed_v2_${parts[*]}"
}

case "${INSTRUMENT}" in
  piano)
    EXPORT_CONFIG="data_piano"
    INSTRUMENT_NAME="salamander_piano"
    INSTRUMENT_PATH="${DEFAULT_PIANO_INSTRUMENT_PATH}"
    BOUNDARY_JSON="${ANALYSIS_DIR}/stats/sfproxy_boundaries/salamander_piano_boundaries.json"
    PITCH_MIN=21
    PITCH_MAX=108
    PITCH_STEP=6
    REGISTER_SPLITS=(48 72)
    DEFAULT_SEGMENTS="${DEFAULT_PIANO_SEGMENTS}"
    ;;
  guitar)
    case "${GUITAR_DATASET}" in
      francoisleduc)
        EXPORT_CONFIG="data_guitar"
        INSTRUMENT_NAME="guitar"
        ;;
      gaps)
        EXPORT_CONFIG="data_guitar_gaps"
        INSTRUMENT_NAME="guitar_gaps"
        ;;
      *)
        echo "Unsupported GUITAR_DATASET='${GUITAR_DATASET}'." >&2
        exit 1
        ;;
    esac
    INSTRUMENT_PATH="${DEFAULT_GUITAR_INSTRUMENT_PATH}"
    BOUNDARY_JSON="${ANALYSIS_DIR}/stats/sfproxy_boundaries/guitar_boundaries.json"
    PITCH_MIN=42
    PITCH_MAX=72
    PITCH_STEP=3
    REGISTER_SPLITS=(52 64)
    DEFAULT_SEGMENTS="${DEFAULT_GUITAR_SEGMENTS}"
    ;;
  *)
    echo "Unsupported INSTRUMENT='${INSTRUMENT}'. Expected 'piano' or 'guitar'." >&2
    exit 1
    ;;
esac

SEGMENT_LIST="$(normalize_list "${SEGMENT_LIST:-${DEFAULT_SEGMENTS}}")"
if [[ -n "${SEGMENT_SECONDS}" ]]; then
  SEGMENT_LIST="$(normalize_list "${SEGMENT_SECONDS}")"
fi

MIXED_COMPONENT_NAMES="$(normalize_list "${MIXED_COMPONENT_NAMES}")"
MIXED_WEIGHTS="$(normalize_list "${MIXED_WEIGHTS}")"
read -r -a mixed_component_names <<< "${MIXED_COMPONENT_NAMES}"
read -r -a mixed_weights <<< "${MIXED_WEIGHTS}"

if [[ "${MIXED_COMPONENT_NAMES}" != "${DEFAULT_MIXED_COMPONENT_NAMES}" ]]; then
  echo "This script expects the four mixed components in order: ${DEFAULT_MIXED_COMPONENT_NAMES}" >&2
  exit 1
fi
if [[ "${#mixed_weights[@]}" -ne 4 ]]; then
  echo "MIXED_WEIGHTS must contain four values in order: boundary coverage realism stress." >&2
  exit 1
fi

MIXED_PRESET_NAME="$(build_mixed_preset_name)"

dataset_dir() {
  local preset="$1"
  local split="$2"
  local segment_seconds="$3"
  echo "${DATA_DIR}/${INSTRUMENT_NAME}/${INSTRUMENT_NAME}_${preset}_$(segment_tag "${segment_seconds}")_${BOUNDARY_MODE}/${split}"
}

dataset_complete() {
  local dir="$1"
  [[ -f "${dir}/configs.pkl" ]] \
    && [[ -f "${dir}/inputs_pitch.pkl" ]] \
    && [[ -f "${dir}/inputs_cont.pkl" ]] \
    && [[ -f "${dir}/inputs_mask.pkl" ]] \
    && [[ -f "${dir}/targets_note.pkl" ]]
}

ensure_boundary_json() {
  local segment_seconds="$1"
  if [[ "${BOUNDARY_MODE}" != "discovered" ]]; then
    return 0
  fi
  if [[ -f "${BOUNDARY_JSON}" ]]; then
    return 0
  fi

  echo "Discovering velocity boundaries -> ${BOUNDARY_JSON}"
  python "${ROOT_DIR}/synth-proxy/src/tools/discover_velocity_boundaries.py" \
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
  local segment_seconds="$1"
  local -n out_ref=$2
  out_ref=()

  case "${BOUNDARY_MODE}" in
    default)
      ;;
    fixed)
      out_ref+=(
        "sampler_options.boundary_v2.velocity_boundary_path=''"
        "sampler_options.boundary_v2.velocity_boundary_strategy=global"
        "sampler_options.coverage_v2.velocity_boundary_path=''"
        "sampler_options.coverage_v2.velocity_boundary_strategy=global"
        "sampler_options.realism_v2.velocity_boundary_path=''"
        "sampler_options.realism_v2.velocity_boundary_strategy=global"
        "sampler_options.stress_v2.velocity_boundary_path=''"
        "sampler_options.stress_v2.velocity_boundary_strategy=global"
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
      ensure_boundary_json "${segment_seconds}"
      out_ref+=(
        "sampler_options.boundary_v2.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.coverage_v2.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.realism_v2.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.stress_v2.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.mixed_v2.components.boundary.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.mixed_v2.components.coverage.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.mixed_v2.components.realism.velocity_boundary_path=${BOUNDARY_JSON}"
        "sampler_options.mixed_v2.components.stress.velocity_boundary_path=${BOUNDARY_JSON}"
      )
      ;;
    *)
      echo "Unsupported BOUNDARY_MODE='${BOUNDARY_MODE}'. Use default, fixed, or discovered." >&2
      exit 1
      ;;
  esac
}

export_split() {
  local preset="$1"
  local split="$2"
  local segment_seconds="$3"
  local out_dir
  local boundary_overrides=()
  out_dir="$(dataset_dir "${preset}" "${split}" "${segment_seconds}")"

  if dataset_complete "${out_dir}"; then
    echo "Reusing existing data: ${out_dir}"
    return 0
  fi

  make_boundary_overrides "${segment_seconds}" boundary_overrides

  python "${ROOT_DIR}/synth-proxy/src/export_dataset_pkl.py" \
    --config-name "${EXPORT_CONFIG}" \
    "paths.repo_root=${ROOT_DIR}" \
    "paths.workspace_dir=${WORKSPACE_BASE}" \
    "paths.analysis_dir=${ANALYSIS_DIR}" \
    "paths.project_name=${PROJECT_NAME}" \
    "instrument.path=${INSTRUMENT_PATH}" \
    "instrument.seg_len_s=${segment_seconds}" \
    "sampler_preset=${preset}" \
    "boundary_mode=${BOUNDARY_MODE}" \
    "split=${split}" \
    "train_dataset_size=${TRAIN_DATASET_SIZE}" \
    "val_dataset_size=${VAL_DATASET_SIZE}" \
    "reset_output_dir=true" \
    "${boundary_overrides[@]}"
}

ensure_preset_exported() {
  local preset="$1"
  local segment_seconds="$2"
  echo "------------------------------------------------------------"
  echo "Prepare data      : ${preset}"
  echo "Instrument        : ${INSTRUMENT_NAME}"
  echo "Segment seconds   : ${segment_seconds}"
  echo "Boundary mode     : ${BOUNDARY_MODE}"
  echo "------------------------------------------------------------"
  export_split "${preset}" train "${segment_seconds}"
  export_split "${preset}" val "${segment_seconds}"
}

ensure_mixed_exported() {
  local segment_seconds="$1"
  local component

  for component in boundary coverage realism stress; do
    ensure_preset_exported "$(component_to_preset "${component}")" "${segment_seconds}"
  done

  local train_out
  local val_out
  train_out="$(dataset_dir "${MIXED_PRESET_NAME}" train "${segment_seconds}")"
  val_out="$(dataset_dir "${MIXED_PRESET_NAME}" val "${segment_seconds}")"

  echo "------------------------------------------------------------"
  echo "Prepare data      : ${MIXED_PRESET_NAME}"
  echo "Mixed components  : ${MIXED_COMPONENT_NAMES}"
  echo "Mixed weights     : ${MIXED_WEIGHTS}"
  echo "------------------------------------------------------------"

  if ! dataset_complete "${train_out}"; then
    mkdir -p "${train_out}"
    python "${ROOT_DIR}/synth-proxy/src/tools/compose_exported_dataset.py" \
      --input-dirs \
      "$(dataset_dir boundary_v2 train "${segment_seconds}")" \
      "$(dataset_dir coverage_v2 train "${segment_seconds}")" \
      "$(dataset_dir realism_v2 train "${segment_seconds}")" \
      "$(dataset_dir stress_v2 train "${segment_seconds}")" \
      --weights "${mixed_weights[@]}" \
      --names boundary coverage realism stress \
      --out-dir "${train_out}" \
      --total-size "${TRAIN_DATASET_SIZE}" \
      --seed 86 \
      --preset-name "${MIXED_PRESET_NAME}"
  else
    echo "Reusing existing data: ${train_out}"
  fi

  if ! dataset_complete "${val_out}"; then
    mkdir -p "${val_out}"
    python "${ROOT_DIR}/synth-proxy/src/tools/compose_exported_dataset.py" \
      --input-dirs \
      "$(dataset_dir boundary_v2 val "${segment_seconds}")" \
      "$(dataset_dir coverage_v2 val "${segment_seconds}")" \
      "$(dataset_dir realism_v2 val "${segment_seconds}")" \
      "$(dataset_dir stress_v2 val "${segment_seconds}")" \
      --weights "${mixed_weights[@]}" \
      --names boundary coverage realism stress \
      --out-dir "${val_out}" \
      --total-size "${VAL_DATASET_SIZE}" \
      --seed 1086 \
      --preset-name "${MIXED_PRESET_NAME}"
  else
    echo "Reusing existing data: ${val_out}"
  fi
}

mkdir -p "${DATA_DIR}" "${ANALYSIS_DIR}/stats/sfproxy_boundaries"
cd "${ROOT_DIR}"

echo "============================================================"
echo "Reusable Data Prep"
echo "Instrument       : ${INSTRUMENT_NAME}"
echo "Segments         : ${SEGMENT_LIST}"
echo "Boundary mode    : ${BOUNDARY_MODE}"
echo "Target preset    : ${TARGET_PRESET}"
if [[ "${TARGET_PRESET}" == "all" ]]; then
  echo "Data families    : coverage_v2 realism_v2 mixed_v2"
fi
if [[ "${TARGET_PRESET}" == "all" || "${TARGET_PRESET}" == "mixed_v2" ]]; then
  echo "Mixed components : ${MIXED_COMPONENT_NAMES}"
  echo "Mixed weights    : ${MIXED_WEIGHTS}"
fi
echo "============================================================"

for segment_seconds in ${SEGMENT_LIST}; do
  echo "============================================================"
  echo "Prepare segment   : ${segment_seconds}s"
  echo "============================================================"

  case "${TARGET_PRESET}" in
    all)
      ensure_preset_exported coverage_v2 "${segment_seconds}"
      ensure_preset_exported realism_v2 "${segment_seconds}"
      ensure_mixed_exported "${segment_seconds}"
      ;;
    coverage_v2|realism_v2)
      ensure_preset_exported "${TARGET_PRESET}" "${segment_seconds}"
      ;;
    mixed_v2)
      ensure_mixed_exported "${segment_seconds}"
      ;;
    *)
      echo "Unsupported TARGET_PRESET='${TARGET_PRESET}'. Use all, coverage_v2, realism_v2, or mixed_v2." >&2
      exit 1
      ;;
  esac
done

echo "Done. Reusable data root: ${DATA_DIR}/${INSTRUMENT_NAME}"
