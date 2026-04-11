#!/bin/bash
#SBATCH --job-name=route3_ablation_pretrain
#SBATCH --output=route3_ablation_pretrain_progress_%A_%a.log
#SBATCH --error=route3_ablation_pretrain_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-63
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_TYPES="${MODEL_TYPES:-hpt_pretrained filmunet_pretrained}"

bash "${SCRIPT_DIR}/kaya_hpt_route3_ablation.sh"
