#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/sfproxy_profile.sh"

WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data"
ANALYSIS_DIR="${ROOT_DIR}/data_analysis"
DATA_DIR="${WORKSPACE_BASE}/synth-proxy/data"

DEFAULT_INSTRUMENT="piano"
DEFAULT_PIANO_DATASET="maestro"
DEFAULT_GUITAR_DATASET="francoisleduc"
DEFAULT_BOUNDARY_MODE="default"
DEFAULT_TARGET_PRESET="all"

INSTRUMENT="${INSTRUMENT:-${DEFAULT_INSTRUMENT}}"
PIANO_DATASET="${PIANO_DATASET:-${DEFAULT_PIANO_DATASET}}"
GUITAR_DATASET="${GUITAR_DATASET:-${DEFAULT_GUITAR_DATASET}}"
BOUNDARY_MODE="${BOUNDARY_MODE:-${DEFAULT_BOUNDARY_MODE}}"
TARGET_PRESET="${TARGET_PRESET:-${DEFAULT_TARGET_PRESET}}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-}"
SEGMENT_LIST="${SEGMENT_LIST:-}"

segment_tag() {
  local value="$1"
  if [[ "${value}" == *.* ]]; then
    value="${value%0}"
    value="${value%.}"
    value="${value//./p}"
  fi
  echo "${value}s"
}

sfproxy_set_profile "${INSTRUMENT}" "${PIANO_DATASET}" "${GUITAR_DATASET}"
REALISM_STATS_JSON="${ANALYSIS_DIR}/stats/midi_sampler/${MIDI_DATASET}_sampler.json"

segment_input="${SEGMENT_SECONDS:-${SEGMENT_LIST:-${DEFAULT_SEGMENTS}}}"
read -r -a SEGMENTS <<< "${segment_input//,/ }"

case "${TARGET_PRESET}" in
  all)
    PRESETS=(coverage_v2 realism_v2 mixed_v2)
    ;;
  boundary_v2|coverage_v2|realism_v2|stress_v2|mixed_v2)
    PRESETS=("${TARGET_PRESET}")
    ;;
  *)
    echo "Unsupported TARGET_PRESET='${TARGET_PRESET}'." >&2
    exit 1
    ;;
esac

mkdir -p "${DATA_DIR}" "${ANALYSIS_DIR}/stats/sfproxy_boundaries"
cd "${ROOT_DIR}"

for segment_seconds in "${SEGMENTS[@]}"; do
  tag="$(segment_tag "${segment_seconds}")"
  boundary_json="${ANALYSIS_DIR}/stats/sfproxy_boundaries/${BOUNDARY_NAME}_boundaries.json"
  boundary_overrides=()

  case "${BOUNDARY_MODE}" in
    default)
      ;;
    fixed)
      boundary_overrides=("v2_boundary_path=''" "v2_boundary_strategy=global")
      ;;
    discovered)
      boundary_json="${ANALYSIS_DIR}/stats/sfproxy_boundaries/${BOUNDARY_NAME}_${tag}_boundaries.json"
      if [[ ! -f "${boundary_json}" ]]; then
        python "${ROOT_DIR}/synth-proxy/src/tools/discover_velocity_boundaries.py" \
          --instrument_path "${INSTRUMENT_PATH}" \
          --instrument_name "${BOUNDARY_NAME}" \
          --bank 0 \
          --program 0 \
          --sr 22050 \
          --seg_len_s "${segment_seconds}" \
          --pitch_min "${PITCH_MIN}" \
          --pitch_max "${PITCH_MAX}" \
          --pitch_step "${PITCH_STEP}" \
          --register_splits "${REGISTER_SPLITS[@]}" \
          --hop 221 \
          --out_json "${boundary_json}"
      fi
      boundary_overrides=("v2_boundary_path=${boundary_json}")
      ;;
    *)
      echo "Unsupported BOUNDARY_MODE='${BOUNDARY_MODE}'. Use default, fixed, or discovered." >&2
      exit 1
      ;;
  esac

  for preset in "${PRESETS[@]}"; do
    case "${preset}" in
      realism_v2|mixed_v2)
        if [[ ! -f "${REALISM_STATS_JSON}" ]]; then
          echo "Missing realism stats: ${REALISM_STATS_JSON}" >&2
          exit 1
        fi
        ;;
    esac

    echo "${preset} ${segment_seconds}s"

    for split in train val; do
      out_dir="${DATA_DIR}/${INSTRUMENT_NAME}/${INSTRUMENT_NAME}_${preset}_${tag}_${BOUNDARY_MODE}/${split}"
      if [[ -f "${out_dir}/configs.pkl" ]] \
        && [[ -f "${out_dir}/inputs_pitch.pkl" ]] \
        && [[ -f "${out_dir}/inputs_cont.pkl" ]] \
        && [[ -f "${out_dir}/inputs_mask.pkl" ]] \
        && [[ -f "${out_dir}/targets_note.pkl" ]]; then
        continue
      fi

      python "${ROOT_DIR}/synth-proxy/src/export_dataset_pkl.py" \
        --config-name "data_${INSTRUMENT}" \
        "paths.repo_root=${ROOT_DIR}" \
        "paths.workspace_dir=${WORKSPACE_BASE}" \
        "paths.analysis_dir=${ANALYSIS_DIR}" \
        "midi_dataset=${MIDI_DATASET}" \
        "instrument.name=${INSTRUMENT_NAME}" \
        "instrument.path=${INSTRUMENT_PATH}" \
        "instrument.seg_len_s=${segment_seconds}" \
        "sampler_preset=${preset}" \
        "boundary_mode=${BOUNDARY_MODE}" \
        "split=${split}" \
        "reset_output_dir=true" \
        "${boundary_overrides[@]}"
    done
  done
done

echo "Done: ${DATA_DIR}/${INSTRUMENT_NAME}"
