#!/usr/bin/env bash
set -euo pipefail

# Manual config.
# Edit these defaults here for the common 5090 workflow.
DEFAULT_MACHINE="5090"            # 3090 or 5090
DEFAULT_SEGMENT_SECONDS="2"       # 2 / 5 / 10
DEFAULT_EPOCHS="7"

SEGMENT_SECONDS="${1:-${DEFAULT_SEGMENT_SECONDS}}"
SEGMENT_TAG="${SEGMENT_SECONDS%.0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${ROOT_DIR}/synthesizer/ddsp-piano-pytorch"
MACHINE="${MACHINE:-${DEFAULT_MACHINE}}"
case "${MACHINE}" in
  3090)
    DEFAULT_MAESTRO_DIR="/media/datadisk/home/22828187/zhanh/Dataset/maestro-v3.0.0"
    DEFAULT_WORKSPACE_BASE="/media/datadisk/home/22828187/zhanh/202601_midisemi_data/ddsp-piano-pytorch"
    ;;
  5090)
    DEFAULT_MAESTRO_DIR="/media/mengh/SharedData/zhanh/Dataset/maestro-v3.0.0"
    DEFAULT_WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data/ddsp-piano-pytorch"
    ;;
  *)
    echo "Unsupported MACHINE='${MACHINE}'. Expected '3090' or '5090'."
    exit 1
    ;;
esac

MAESTRO_DIR="${MAESTRO_DIR:-${DEFAULT_MAESTRO_DIR}}"
WORKSPACE_BASE="${WORKSPACE_BASE:-${DEFAULT_WORKSPACE_BASE}}"
WORKSPACE_DIR="${WORKSPACE_BASE}/workspaces_unified_${SEGMENT_TAG}s"
CACHE_DIR="${WORKSPACE_DIR}/data_cache"
EXP_DIR="${WORKSPACE_DIR}/models"

PYTHON_BIN="${PYTHON_BIN:-/home/mengh/miniconda3/envs/dynest_ddsp/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-ddsp-piano-unified}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
LR="${LR:-0.001}"

mkdir -p "${CACHE_DIR}" "${EXP_DIR}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -f "${MAESTRO_DIR}/maestro-v3.0.0.csv" ]; then
  echo "MAESTRO metadata file not found under: ${MAESTRO_DIR}" >&2
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util, sys
missing = [m for m in ("note_seq", "torch", "wandb", "numpy") if importlib.util.find_spec(m) is None]
if missing:
    print(f"Missing Python modules: {missing}", file=sys.stderr)
    sys.exit(1)
PY

cd "${PROJECT_DIR}"

if [ ! -f "${CACHE_DIR}/cache_config.json" ] || [ ! -d "${CACHE_DIR}/train" ] || [ ! -d "${CACHE_DIR}/validation" ] || [ ! -d "${CACHE_DIR}/test" ]; then
  echo "Prepared piano cache not found. Running preprocessing first."
  "${PYTHON_BIN}" preprocess.py \
    --splits train validation test \
    --segment_duration "${SEGMENT_SECONDS}" \
    --sample_rate 22050 \
    --frame_rate 100 \
    --max_polyphony 16 \
    "${MAESTRO_DIR}" \
    "${CACHE_DIR}"
else
  echo "Prepared piano cache found. Skipping preprocessing."
fi

"${PYTHON_BIN}" train.py \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --phase 1 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration "${SEGMENT_SECONDS}" \
  --save_interval "${CHECKPOINT_INTERVAL}" \
  --num_workers "${NUM_WORKERS}" \
  "${CACHE_DIR}" \
  "${EXP_DIR}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "phase1_${SEGMENT_TAG}s"
