#!/bin/bash
#SBATCH --job-name=scoreinf_proxy_audio
#SBATCH --output=scoreinf_proxy_audio_%A_%a.log
#SBATCH --error=scoreinf_proxy_audio_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

# Proxy-audio ablation for weakly supervised velocity training.
#
# Experiments:
#   0  A_supervised_only
#   1  B_proxy_mstft
#   2  C_proxy_piano_ssm_spectral
#   3  D_proxy_piano_ssm_spectral_plus_log_rms   <- new default
#   4  E_proxy_piano_ssm_combined
#   5  F_proxy_piano_ssm_combined_rm
#   6  G_proxy_piano_ssm_spectral_plus_ddsp_loudness
#
# Required before running proxy-enabled jobs:
#   export PROXY_CKPT=/absolute/path/to/ddsp_piano_checkpoint.pth
#   export PROXY_ROOT=/absolute/path/to/ddsp-piano-pytorch
#
# Optional overrides:
#   export MODEL_TYPE=hpt
#   export SCORE_METHOD=direct
#   export INPUT2=null
#   export INPUT3=null
#   export LOSS_TYPE=kim_bce_l1
#   export PROXY_WEIGHT=1.0
#   export LOGRMS_WEIGHT=0.05
#   export DDSP_LOUDNESS_WEIGHT=0.05
#   export PROJECT_NAME=202510_hpt_smc
#
# Launch:
#   sbatch run_scripts/kaya_hpt_proxy_audio_ablation.sh

module load Anaconda3/2024.06 gcc/11.5.0 cuda/12.4.1
module list
source activate bark_env

set -euo pipefail

echo "Running on host: $(hostname)"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM ID: ${SLURM_ARRAY_ID:-N/A} ${SLURM_ARRAY_TASK_ID:-N/A}"

FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PROJECT_NAME=${PROJECT_NAME:-202510_hpt_smc}
EXECUTABLE=$HOME/${PROJECT_NAME}
SCRATCH=$MYSCRATCH/${PROJECT_NAME}/$FOLDER_NAME
RESULTS=$MYGROUP/${PROJECT_NAME}_results/$FOLDER_NAME

mkdir -p "$SCRATCH" "$RESULTS"
echo "SCRATCH is $SCRATCH"
echo "RESULTS dir is $RESULTS"

echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r "$EXECUTABLE" "$SCRATCH"
cd "$SCRATCH/$PROJECT_NAME"

WORKSPACE_DIR="$SCRATCH/$PROJECT_NAME/workspaces"
mkdir -p "$WORKSPACE_DIR"

DATA_SRC=$MYSCRATCH/202510_hpt_data/workspaces/hdf5s
DATA_VIEW=$WORKSPACE_DIR/hdf5s
rm -rf "$DATA_VIEW"
ln -s "$DATA_SRC" "$DATA_VIEW"

MODEL_TYPE=${MODEL_TYPE:-hpt}
SCORE_METHOD=${SCORE_METHOD:-direct}
INPUT2=${INPUT2:-null}
INPUT3=${INPUT3:-null}
LOSS_TYPE=${LOSS_TYPE:-kim_bce_l1}
PROXY_TYPE=${PROXY_TYPE:-ddsp_piano}
PROXY_WEIGHT=${PROXY_WEIGHT:-1.0}
LOGRMS_WEIGHT=${LOGRMS_WEIGHT:-0.05}
DDSP_LOUDNESS_WEIGHT=${DDSP_LOUDNESS_WEIGHT:-0.05}
PROXY_CKPT=${PROXY_CKPT:-}
PROXY_ROOT=${PROXY_ROOT:-}

EXP_NAME=(
  "A_supervised_only"
  "B_proxy_mstft"
  "C_proxy_piano_ssm_spectral"
  "D_proxy_piano_ssm_spectral_plus_log_rms"
  "E_proxy_piano_ssm_combined"
  "F_proxy_piano_ssm_combined_rm"
  "G_proxy_piano_ssm_spectral_plus_ddsp_loudness"
)
EXP_PROXY_ENABLED=(
  "false"
  "true"
  "true"
  "true"
  "true"
  "true"
  "true"
)
EXP_PROXY_WEIGHT=(
  "0.0"
  "$PROXY_WEIGHT"
  "$PROXY_WEIGHT"
  "$PROXY_WEIGHT"
  "$PROXY_WEIGHT"
  "$PROXY_WEIGHT"
  "$PROXY_WEIGHT"
)
EXP_AUDIO_LOSS=(
  "multi_scale_stft"
  "multi_scale_stft"
  "piano_ssm_spectral"
  "piano_ssm_spectral_plus_log_rms"
  "piano_ssm_combined"
  "piano_ssm_combined_rm"
  "piano_ssm_spectral_plus_ddsp_loudness"
)

TOTAL_JOBS=${#EXP_NAME[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((TOTAL_JOBS-1)))"
  exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
EXP_TAG=${EXP_NAME[$IDX]}
CUR_PROXY_ENABLED=${EXP_PROXY_ENABLED[$IDX]}
CUR_PROXY_WEIGHT=${EXP_PROXY_WEIGHT[$IDX]}
CUR_AUDIO_LOSS=${EXP_AUDIO_LOSS[$IDX]}

if [ "$CUR_PROXY_ENABLED" = "true" ]; then
  if [ -z "$PROXY_CKPT" ]; then
    echo "Error: PROXY_CKPT is empty, but experiment '$EXP_TAG' requires proxy training." >&2
    exit 1
  fi
  if [ -z "$PROXY_ROOT" ]; then
    echo "Error: PROXY_ROOT is empty, but experiment '$EXP_TAG' requires the DDSP proxy repo." >&2
    exit 1
  fi
fi

echo "Experiment       : $EXP_TAG"
echo "Model            : $MODEL_TYPE"
echo "Score method     : $SCORE_METHOD"
echo "Input2 / Input3  : $INPUT2 / $INPUT3"
echo "Velocity loss    : $LOSS_TYPE"
echo "Proxy enabled    : $CUR_PROXY_ENABLED"
echo "Proxy type       : $PROXY_TYPE"
echo "Proxy audio loss : $CUR_AUDIO_LOSS"
echo "Proxy weight     : $CUR_PROXY_WEIGHT"
echo "LOGRMS weight    : $LOGRMS_WEIGHT"
echo "DDSP loud weight : $DDSP_LOUDNESS_WEIGHT"

python pytorch/train_proxy.py \
  exp.workspace="$WORKSPACE_DIR" \
  model.type="$MODEL_TYPE" \
  score_informed.method="$SCORE_METHOD" \
  model.input2="$INPUT2" \
  model.input3="$INPUT3" \
  loss.loss_type="$LOSS_TYPE" \
  loss.supervised_weight=1.0 \
  loss.proxy_weight="$CUR_PROXY_WEIGHT" \
  proxy.enabled="$CUR_PROXY_ENABLED" \
  proxy.type="$PROXY_TYPE" \
  proxy.checkpoint="$PROXY_CKPT" \
  proxy.project_root="$PROXY_ROOT" \
  proxy.audio_loss.type="$CUR_AUDIO_LOSS" \
  proxy.audio_loss.piano_ssm_spectral_plus_log_rms.log_rms_weight="$LOGRMS_WEIGHT" \
  proxy.audio_loss.piano_ssm_spectral_plus_ddsp_loudness.loudness_weight="$DDSP_LOUDNESS_WEIGHT" \
  wandb.comment="$EXP_TAG"

[ -d "$WORKSPACE_DIR/checkpoints" ] && mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"

cd "$HOME"
rm -r "$SCRATCH"
conda deactivate 2>/dev/null || deactivate 2>/dev/null || true
echo scoreinf_proxy_audio $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
