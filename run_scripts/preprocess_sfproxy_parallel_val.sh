#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/sfproxy_profile.sh"

WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data"
ANALYSIS_DIR="${ROOT_DIR}/data_analysis"
DATA_DIR="${WORKSPACE_BASE}/synth-proxy/data"
LOG_DIR="${SCRIPT_DIR}/logs"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

INSTRUMENT="${INSTRUMENT:-piano}"
PIANO_DATASET="${PIANO_DATASET:-maestro}"
GUITAR_DATASET="${GUITAR_DATASET:-francoisleduc}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-2}"
BOUNDARY_MODE="${BOUNDARY_MODE:-default}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-5}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-4}"
EXPORT_PREFETCH_FACTOR="${EXPORT_PREFETCH_FACTOR:-4}"
MIX_WEIGHTS="${MIX_WEIGHTS:-}"

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
if [[ ! -f "${REALISM_STATS_JSON}" ]]; then
  echo "Missing realism stats: ${REALISM_STATS_JSON}" >&2
  exit 1
fi

tag="$(segment_tag "${SEGMENT_SECONDS}")"
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
        --seg_len_s "${SEGMENT_SECONDS}" \
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
    echo "Unsupported BOUNDARY_MODE='${BOUNDARY_MODE}'." >&2
    exit 1
    ;;
esac

mix_weight_args=()
if [[ -n "${MIX_WEIGHTS}" ]]; then
  read -r -a mix_weights <<< "${MIX_WEIGHTS//,/ }"
  if [[ "${#mix_weights[@]}" -ne 4 ]]; then
    echo "MIX_WEIGHTS must provide 4 values: boundary coverage realism stress" >&2
    exit 1
  fi
  mix_weight_args=(
    --names boundary coverage realism stress
    --weights "${mix_weights[@]}"
  )
fi

launch_export() {
  local preset="$1"
  local out_dir="${DATA_DIR}/${INSTRUMENT_NAME}/${preset}_${tag}_${BOUNDARY_MODE}/val"
  if [[ -f "${out_dir}/configs.pkl" ]] \
    && [[ -f "${out_dir}/inputs_pitch.pkl" ]] \
    && [[ -f "${out_dir}/inputs_cont.pkl" ]] \
    && [[ -f "${out_dir}/inputs_mask.pkl" ]] \
    && [[ -f "${out_dir}/targets_note.pkl" ]]; then
    echo "skip ${preset} val"
    return 0
  fi

  python "${ROOT_DIR}/synth-proxy/src/export_dataset_pkl.py" \
    --config-name "data_${INSTRUMENT}" \
    "paths.repo_root=${ROOT_DIR}" \
    "paths.workspace_dir=${WORKSPACE_BASE}" \
    "paths.analysis_dir=${ANALYSIS_DIR}" \
    "midi_dataset=${MIDI_DATASET}" \
    "instrument.name=${INSTRUMENT_NAME}" \
    "instrument.path=${INSTRUMENT_PATH}" \
    "instrument.seg_len_s=${SEGMENT_SECONDS}" \
    "sampler_preset=${preset}" \
    "boundary_mode=${BOUNDARY_MODE}" \
    "split=val" \
    "reset_output_dir=true" \
    "batch_size=${EXPORT_BATCH_SIZE}" \
    "num_workers=${EXPORT_NUM_WORKERS}" \
    "prefetch_factor=${EXPORT_PREFETCH_FACTOR}" \
    "${boundary_overrides[@]}"
}

mkdir -p "${DATA_DIR}" "${ANALYSIS_DIR}/stats/sfproxy_boundaries"
mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}/synth-proxy"

pids=()
for preset in boundary_v2 coverage_v2 realism_v2 stress_v2; do
  log_path="${LOG_DIR}/${INSTRUMENT_NAME}_${preset}_${tag}_${BOUNDARY_MODE}_val_${RUN_STAMP}.log"
  echo "launch ${preset} val -> ${log_path}"
  launch_export "${preset}" > "${log_path}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

out_dir="${DATA_DIR}/${INSTRUMENT_NAME}/mixed_v2_${tag}_${BOUNDARY_MODE}/val"
rm -rf "${out_dir}"
compose_log="${LOG_DIR}/${INSTRUMENT_NAME}_mixed_v2_${tag}_${BOUNDARY_MODE}_val_${RUN_STAMP}.log"

python "${ROOT_DIR}/synth-proxy/src/tools/compose_exported_dataset.py" \
  --input-dirs \
    "${DATA_DIR}/${INSTRUMENT_NAME}/boundary_v2_${tag}_${BOUNDARY_MODE}/val" \
    "${DATA_DIR}/${INSTRUMENT_NAME}/coverage_v2_${tag}_${BOUNDARY_MODE}/val" \
    "${DATA_DIR}/${INSTRUMENT_NAME}/realism_v2_${tag}_${BOUNDARY_MODE}/val" \
    "${DATA_DIR}/${INSTRUMENT_NAME}/stress_v2_${tag}_${BOUNDARY_MODE}/val" \
  --config-file "${ROOT_DIR}/synth-proxy/configs/data_${INSTRUMENT}.yaml" \
  --preset mixed_v2 \
  --preset-name mixed_v2 \
  "${mix_weight_args[@]}" \
  --out-dir "${out_dir}" > "${compose_log}" 2>&1

echo "Done val: ${out_dir}"
echo "Logs: ${LOG_DIR}"
