#!/usr/bin/env bash
set -euo pipefail

# Manual config.
DEFAULT_GUITAR_DATASET="francoisleduc"    # francoisleduc or gaps

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${ROOT_DIR}/score_hpt"
DDSP_PROJECT_ROOT="${ROOT_DIR}/synthesizer/ddsp-guitar-synth"
PYTHON_BIN="${PYTHON_BIN:-python}"
GUITAR_DATASET="${GUITAR_DATASET:-${DEFAULT_GUITAR_DATASET}}"

DEFAULT_FRANCOISLEDUC_DIR="/media/datadisk/home/22828187/zhanh/Dataset/FrancoisLeducGuitarDataset"
DEFAULT_GAPS_DIR="/media/datadisk/home/22828187/zhanh/Dataset/GAPS"
DEFAULT_WORKSPACE_BASE="/media/datadisk/home/22828187/zhanh/202601_midisemi_data"
DEFAULT_DDSP_CKPT="/media/datadisk/home/22828187/zhanh/202601_midisemi_data/ddsp-guitar-synth/flgd_5s/output/ddsp_guitar_synth_sr22050_fps100_seg5s/latest_model_checkpoint.pt"

case "${GUITAR_DATASET}" in
  francoisleduc)
    DEFAULT_GUITAR_DATASET_DIR="${DEFAULT_FRANCOISLEDUC_DIR}"
    PACK_MODE="pack_francoisleduc_dataset_to_hdf5"
    HDF5_DIR_NAME="francoisleduc_sr22050"
    DATASET_DIR_OVERRIDE_KEY="dataset.francoisleduc_dir"
    ;;
  gaps)
    DEFAULT_GUITAR_DATASET_DIR="${DEFAULT_GAPS_DIR}"
    PACK_MODE="pack_gaps_dataset_to_hdf5"
    HDF5_DIR_NAME="gaps_sr22050"
    DATASET_DIR_OVERRIDE_KEY="dataset.gaps_dir"
    ;;
  *)
    echo "Unsupported GUITAR_DATASET='${GUITAR_DATASET}'. Expected 'francoisleduc' or 'gaps'." >&2
    exit 1
    ;;
esac

GUITAR_DATASET_DIR="${GUITAR_DATASET_DIR:-${DEFAULT_GUITAR_DATASET_DIR}}"
WORKSPACE_BASE="${WORKSPACE_BASE:-${DEFAULT_WORKSPACE_BASE}}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_BASE}/score_hpt/workspaces}"
DDSP_CKPT="${DDSP_CKPT:-${DEFAULT_DDSP_CKPT}}"
HDF5_DIR="${WORKSPACE_DIR}/hdf5s/${HDF5_DIR_NAME}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
PROXY_WEIGHT="${PROXY_WEIGHT:-1.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.0}"

mkdir -p "${WORKSPACE_DIR}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -d "${DDSP_PROJECT_ROOT}" ]; then
  echo "DDSP project root not found: ${DDSP_PROJECT_ROOT}" >&2
  exit 1
fi

if [ ! -d "${GUITAR_DATASET_DIR}" ]; then
  echo "Dataset directory not found: ${GUITAR_DATASET_DIR}" >&2
  exit 1
fi

if [ ! -f "${DDSP_CKPT}" ]; then
  echo "DDSP checkpoint not found: ${DDSP_CKPT}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} not found in PATH. Please activate the correct environment first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [ ! -d "${HDF5_DIR}" ] || ! compgen -G "${HDF5_DIR}/*.h5" >/dev/null; then
  echo "Packed HDF5 dataset not found. Running preprocessing first."
  "${PYTHON_BIN}" pytorch/data_generator.py "${PACK_MODE}" \
    "exp.workspace=${WORKSPACE_DIR}" \
    "feature.sample_rate=22050" \
    "${DATASET_DIR_OVERRIDE_KEY}=${GUITAR_DATASET_DIR}"
else
  echo "Packed HDF5 dataset found. Skipping preprocessing."
fi

"${PYTHON_BIN}" pytorch/train_ddsp.py \
  "exp.workspace=${WORKSPACE_DIR}" \
  "exp.batch_size=4" \
  "dataset.train_set=${GUITAR_DATASET}" \
  "dataset.test_set=${GUITAR_DATASET}" \
  "dataset.eval_sets=[train,${GUITAR_DATASET}]" \
  "model.type=hpt" \
  "model.input2=onset" \
  "model.input3=frame" \
  "score_informed.method=note_editor" \
  "loss.supervised_weight=${SUPERVISED_WEIGHT}" \
  "loss.proxy_weight=${PROXY_WEIGHT}" \
  "loss.velocity_prior_weight=${PRIOR_WEIGHT}" \
  "proxy.enabled=true" \
  "proxy.type=diffsynth_guitar" \
  "proxy.project_root=${DDSP_PROJECT_ROOT}" \
  "proxy.checkpoint=${DDSP_CKPT}" \
  "proxy.backend_segment_seconds=0.0" \
  "wandb.comment=route3_guitar_${GUITAR_DATASET}"
