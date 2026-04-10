#!/usr/bin/env bash

score_hpt_segment_tag() {
  local value="${1%.0}"
  printf '%ss\n' "${value}"
}

score_hpt_has_hdf5() {
  find "$1" -type f -name '*.h5' -print -quit 2>/dev/null | grep -q .
}

score_hpt_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

score_hpt_strip_yaml_value() {
  local value="${1%%#*}"
  value="$(score_hpt_trim "${value}")"
  value="${value#\"}"
  value="${value%\"}"
  value="${value#\'}"
  value="${value%\'}"
  printf '%s\n' "${value}"
}

score_hpt_yaml_get() {
  local section="$1"
  local key="$2"
  local path="$3"

  awk -v section="${section}" -v key="${key}" '
    /^[^[:space:]#][^:]*:/ {
      name = $1
      sub(/:.*/, "", name)
      in_section = (name == section)
    }

    in_section {
      pattern = "^[[:space:]]*" key ":[[:space:]]*"
      if ($0 ~ pattern) {
        sub(pattern, "", $0)
        print
        exit
      }
    }
  ' "${path}"
}

score_hpt_abspath() {
  local base="$1"
  local path="$2"
  local dir
  local name

  case "${path}" in
    /*)
      printf '%s\n' "${path}"
      ;;
    *)
      if command -v realpath >/dev/null 2>&1; then
        realpath -m "${base}/${path}"
        return
      fi
      dir="$(dirname "${base}/${path}")"
      name="$(basename "${path}")"
      (
        cd "${dir}" 2>/dev/null &&
        printf '%s/%s\n' "$PWD" "${name}"
      ) || printf '%s/%s\n' "${base}" "${path}"
      ;;
  esac
}

score_hpt_workspace_data_root() {
  local workspace_dir="$1"
  local resolved="${workspace_dir}"

  if [ -e "${workspace_dir}" ] && command -v realpath >/dev/null 2>&1; then
    resolved="$(realpath "${workspace_dir}")"
  fi

  dirname "$(dirname "${resolved}")"
}

score_hpt_config_value() {
  local section="$1"
  local key="$2"
  score_hpt_strip_yaml_value "$(score_hpt_yaml_get "${section}" "${key}" "${CONFIG_PATH}")"
}

score_hpt_init_context() {
  PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}/score_hpt}"
  CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/pytorch/config/config.yaml}"

  if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Score-HPT config not found: ${CONFIG_PATH}" >&2
    exit 1
  fi

  local workspace_rel
  workspace_rel="$(score_hpt_config_value exp workspace)"
  WORKSPACE_DIR="$(score_hpt_abspath "${PROJECT_DIR}" "${WORKSPACE_DIR:-${workspace_rel}}")"
  DATA_ROOT="${DATA_ROOT:-$(score_hpt_workspace_data_root "${WORKSPACE_DIR}")}"
}

score_hpt_set_dataset_profile() {
  local dataset_name="$1"
  local dataset_dir_rel

  case "${dataset_name}" in
    maestro)
      DATASET_KIND="piano"
      PACK_MODE="pack_maestro_dataset_to_hdf5"
      PACK_OVERRIDE_KEY="dataset.maestro_dir"
      HDF5_DIR_NAME="maestro_sr22050"
      DATASET_REQUIRED_PATH="maestro-v3.0.0.csv"
      DEFAULT_TEST_SET="maestro"
      DEFAULT_EVAL_SETS="[train,maestro,smd]"
      DIFFSYNTH_PROXY_TYPE="diffsynth_piano"
      DDSP_PROJECT_ROOT_DEFAULT="${ROOT_DIR}/synthesizer/ddsp-piano-pytorch"
      DDSP_CKPT_ROOT_DEFAULT="${DATA_ROOT}/ddsp-piano-pytorch"
      SFPROXY_DATASET_NAME_DEFAULT="piano"
      SFPROXY_INSTRUMENT_NAME_DEFAULT="salamander_piano"
      DATASET_DIR_ENV_VALUE="${MAESTRO_DIR:-}"
      dataset_dir_rel="$(score_hpt_config_value dataset maestro_dir)"
      ;;
    smd)
      DATASET_KIND="piano"
      PACK_MODE="pack_smd_dataset_to_hdf5"
      PACK_OVERRIDE_KEY="dataset.smd_dir"
      HDF5_DIR_NAME="smd_sr22050"
      DATASET_REQUIRED_PATH=""
      DEFAULT_TEST_SET="smd"
      DEFAULT_EVAL_SETS="[train,smd]"
      DIFFSYNTH_PROXY_TYPE="diffsynth_piano"
      DDSP_PROJECT_ROOT_DEFAULT="${ROOT_DIR}/synthesizer/ddsp-piano-pytorch"
      DDSP_CKPT_ROOT_DEFAULT="${DATA_ROOT}/ddsp-piano-pytorch"
      SFPROXY_DATASET_NAME_DEFAULT="piano_smd"
      SFPROXY_INSTRUMENT_NAME_DEFAULT="salamander_piano_smd"
      DATASET_DIR_ENV_VALUE="${SMD_DIR:-}"
      dataset_dir_rel="$(score_hpt_config_value dataset smd_dir)"
      ;;
    francoisleduc)
      DATASET_KIND="guitar"
      PACK_MODE="pack_francoisleduc_dataset_to_hdf5"
      PACK_OVERRIDE_KEY="dataset.francoisleduc_dir"
      HDF5_DIR_NAME="francoisleduc_sr22050"
      DATASET_REQUIRED_PATH="metadata.csv"
      DEFAULT_TEST_SET="francoisleduc"
      DEFAULT_EVAL_SETS="[train,francoisleduc]"
      DIFFSYNTH_PROXY_TYPE="diffsynth_guitar"
      DDSP_PROJECT_ROOT_DEFAULT="${ROOT_DIR}/synthesizer/ddsp-guitar-synth"
      DDSP_CKPT_ROOT_DEFAULT="${DATA_ROOT}/ddsp-guitar-synth"
      SFPROXY_DATASET_NAME_DEFAULT="guitar"
      SFPROXY_INSTRUMENT_NAME_DEFAULT="guitar"
      DATASET_DIR_ENV_VALUE="${FRANCOISLEDUC_DIR:-${GUITAR_DATASET_DIR:-}}"
      dataset_dir_rel="$(score_hpt_config_value dataset francoisleduc_dir)"
      ;;
    gaps)
      DATASET_KIND="guitar"
      PACK_MODE="pack_gaps_dataset_to_hdf5"
      PACK_OVERRIDE_KEY="dataset.gaps_dir"
      HDF5_DIR_NAME="gaps_sr22050"
      DATASET_REQUIRED_PATH="gaps_metadata_with_splits.csv"
      DEFAULT_TEST_SET="gaps"
      DEFAULT_EVAL_SETS="[train,gaps]"
      DIFFSYNTH_PROXY_TYPE="diffsynth_guitar"
      DDSP_PROJECT_ROOT_DEFAULT="${ROOT_DIR}/synthesizer/ddsp-guitar-synth"
      DDSP_CKPT_ROOT_DEFAULT="${DATA_ROOT}/ddsp-guitar-synth"
      SFPROXY_DATASET_NAME_DEFAULT="guitar_gaps"
      SFPROXY_INSTRUMENT_NAME_DEFAULT="guitar_gaps"
      DATASET_DIR_ENV_VALUE="${GAPS_DIR:-${GUITAR_DATASET_DIR:-}}"
      dataset_dir_rel="$(score_hpt_config_value dataset gaps_dir)"
      ;;
    *)
      echo "Unsupported TRAIN_SET='${dataset_name}'. Expected maestro, smd, francoisleduc, or gaps." >&2
      exit 1
      ;;
  esac

  DATASET_NAME="${dataset_name}"
  DATASET_DIR="$(score_hpt_abspath "${PROJECT_DIR}" "${DATASET_DIR_ENV_VALUE:-${dataset_dir_rel}}")"
  HDF5_DIR="${WORKSPACE_DIR}/hdf5s/${HDF5_DIR_NAME}"
}

score_hpt_assert_dataset_dir() {
  if [ -n "${DATASET_REQUIRED_PATH}" ]; then
    if [ ! -f "${DATASET_DIR}/${DATASET_REQUIRED_PATH}" ]; then
      echo "Dataset metadata not found under: ${DATASET_DIR}" >&2
      exit 1
    fi
    return
  fi

  if [ ! -d "${DATASET_DIR}" ]; then
    echo "Dataset directory not found: ${DATASET_DIR}" >&2
    exit 1
  fi
}

score_hpt_ensure_dataset_hdf5() {
  local python_bin="$1"

  if [ ! -d "${HDF5_DIR}" ] || ! score_hpt_has_hdf5 "${HDF5_DIR}"; then
    echo "Packed ${DATASET_NAME} HDF5 dataset not found. Running preprocessing first."
    "${python_bin}" pytorch/data_generator.py "${PACK_MODE}" \
      "exp.workspace=${WORKSPACE_DIR}" \
      "feature.sample_rate=22050" \
      "${PACK_OVERRIDE_KEY}=${DATASET_DIR}"
  fi
}

score_hpt_collect_eval_datasets() {
  local eval_sets="${1:-}"
  local item

  eval_sets="${eval_sets#[}"
  eval_sets="${eval_sets%]}"
  eval_sets="${eval_sets//,/ }"

  for item in ${eval_sets}; do
    item="${item//\"/}"
    item="${item//\'/}"
    item="$(score_hpt_trim "${item}")"
    [ -n "${item}" ] && [ "${item}" != "train" ] && printf '%s\n' "${item}"
  done
}

score_hpt_prepare_required_datasets() {
  local python_bin="$1"
  shift

  local dataset_name
  local seen=" "

  for dataset_name in "$@"; do
    [ -n "${dataset_name}" ] || continue
    case " ${seen} " in
      *" ${dataset_name} "*) continue ;;
    esac
    seen="${seen}${dataset_name} "
    score_hpt_set_dataset_profile "${dataset_name}"
    score_hpt_assert_dataset_dir
    score_hpt_ensure_dataset_hdf5 "${python_bin}"
  done
}

score_hpt_resolve_ddsp_ckpt() {
  local dataset_name="$1"
  local segment="$2"
  local seg_tag
  local ckpt_root
  local ddsp_phase
  local ckpt_epoch
  local path

  seg_tag="$(score_hpt_segment_tag "${segment}")"
  score_hpt_set_dataset_profile "${dataset_name}"
  ckpt_root="${DDSP_CKPT_ROOT:-${DDSP_CKPT_ROOT_DEFAULT}}"
  ddsp_phase="${DDSP_PHASE:-1}"
  ckpt_epoch="${CKPT_EPOCH:-7}"

  case "${DATASET_KIND}" in
    piano)
      path="${ckpt_root}/workspaces_unified_${seg_tag}/models/phase_${ddsp_phase}/ckpts/ddsp-piano_epoch_${ckpt_epoch}_params.pt"
      ;;
    guitar)
      path="${ckpt_root}/${dataset_name}_${seg_tag}/output/latest_model_checkpoint.pt"
      ;;
  esac

  if [ -f "${path}" ]; then
    printf '%s\n' "${path}"
  fi
}
