#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <segment_seconds> <francoisledu_path> <output_dir>"
  exit 1
fi

SEGMENT_SECONDS="$1"
FRANCOISLEDU_PATH="$2"
OUTPUT_DIR="$3"

python prepare_flgd_unified.py \
  --segment_seconds "${SEGMENT_SECONDS}" \
  --francoisledu_path "${FRANCOISLEDU_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_rate 22050 \
  --frame_rate 100
