#!/usr/bin/env bash
set -euo pipefail

# Manual config.
DEFAULT_MACHINE="3090"            # 3090 or 5090
DEFAULT_SEGMENT_SECONDS="2"       # 2 / 5 / 10

SEGMENT_SECONDS="${1:-${DEFAULT_SEGMENT_SECONDS}}"
SEGMENT_TAG="${SEGMENT_SECONDS%.0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${ROOT_DIR}/synthesizer/ddsp-guitar-synth"
MACHINE="${MACHINE:-${DEFAULT_MACHINE}}"

case "${MACHINE}" in
  3090)
    DEFAULT_FRANCOISLEDU_DIR="/media/datadisk/home/22828187/zhanh/Dataset/FrancoisLeducGuitarDataset"
    DEFAULT_WORKSPACE_BASE="/media/datadisk/home/22828187/zhanh/202601_midisemi_data/ddsp-guitar-synth"
    ;;
  5090)
    DEFAULT_FRANCOISLEDU_DIR="/media/mengh/SharedData/zhanh/Dataset/FrancoisLeducGuitarDataset"
    DEFAULT_WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data/ddsp-guitar-synth"
    ;;
  *)
    echo "Unsupported MACHINE='${MACHINE}'. Expected '3090' or '5090'."
    exit 1
    ;;
esac

FRANCOISLEDU_DIR="${FRANCOISLEDU_DIR:-${DEFAULT_FRANCOISLEDU_DIR}}"
WORKSPACE_BASE="${WORKSPACE_BASE:-${DEFAULT_WORKSPACE_BASE}}"
RUN_DIR="${WORKSPACE_BASE}/flgd_${SEGMENT_TAG}s"
DATA_DIR="${RUN_DIR}/data"
OUTPUT_DIR="${RUN_DIR}/output"
TRAIN_DATASET="${DATA_DIR}/train_flgd_midi_${SEGMENT_TAG}s.npz"
VAL_DATASET="${DATA_DIR}/val_flgd_midi_${SEGMENT_TAG}s.npz"

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -f "${FRANCOISLEDU_DIR}/metadata.csv" ]; then
  echo "FrancoisLeduc metadata.csv not found under: ${FRANCOISLEDU_DIR}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH. Please activate the correct environment first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [ ! -f "${TRAIN_DATASET}" ] || [ ! -f "${VAL_DATASET}" ]; then
  echo "Prepared datasets not found. Running data preparation first."
  python prepare_flgd_unified.py \
    --segment_seconds "${SEGMENT_SECONDS}" \
    --francoisledu_path "${FRANCOISLEDU_DIR}" \
    --output_dir "${DATA_DIR}" \
    --sample_rate 22050 \
    --frame_rate 100
else
  echo "Prepared datasets found. Skipping data preparation."
fi

python train_midi_synth_unified.py \
  --train_dataset_path "${TRAIN_DATASET}" \
  --val_dataset_path "${VAL_DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds "${SEGMENT_SECONDS}" \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
