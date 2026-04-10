#!/bin/bash
#SBATCH --job-name=route3_ablation
#SBATCH --output=route3_ablation_progress_%A_%a.log
#SBATCH --error=route3_ablation_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

module load Anaconda3/2024.06 cuda/12.4.1 gcc/12.4.0
module list
source activate bark_env

set -euo pipefail

echo "Running on host: $(hostname)"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM ID: ${SLURM_ARRAY_ID:-N/A} ${SLURM_ARRAY_TASK_ID:-N/A}"

FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PROJECT_NAME=${PROJECT_NAME:-202604_midiproxy}
DATA_PROJECT=${DATA_PROJECT:-202604_midiproxy_data}
EXECUTABLE=$HOME/${PROJECT_NAME}
SCRATCH=$MYSCRATCH/${PROJECT_NAME}/$FOLDER_NAME
RESULTS=$MYGROUP/${PROJECT_NAME}_results/$FOLDER_NAME

mkdir -p "$SCRATCH" "$RESULTS"
echo "SCRATCH is $SCRATCH"
echo "RESULTS dir is $RESULTS"

echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r "$EXECUTABLE" "$SCRATCH"
cd "$SCRATCH/$PROJECT_NAME/score_hpt"

WORKSPACE_DIR=workspaces
rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"

DATA_ROOT="$MYSCRATCH/${DATA_PROJECT}"
KAYA_DATA_DIR=../kaya_data
rm -rf "$KAYA_DATA_DIR"
mkdir -p "$KAYA_DATA_DIR"

HDF5_SRC="$DATA_ROOT/score_hpt/workspaces/hdf5s"
HDF5_VIEW="$WORKSPACE_DIR/hdf5s"
DDSP_SRC="$DATA_ROOT/ddsp-piano-pytorch"
DDSP_VIEW="$KAYA_DATA_DIR/ddsp-piano-pytorch"
SFPROXY_SRC="$DATA_ROOT/synth-proxy"
SFPROXY_VIEW="$KAYA_DATA_DIR/synth-proxy"

ln -s "$HDF5_SRC" "$HDF5_VIEW"
ln -s "$DDSP_SRC" "$DDSP_VIEW"
ln -s "$SFPROXY_SRC" "$SFPROXY_VIEW"

[ -d "$HDF5_VIEW/maestro_sr22050" ] || { echo "Missing MAESTRO HDF5: $HDF5_VIEW/maestro_sr22050" >&2; exit 1; }
[ -d "$HDF5_VIEW/smd_sr22050" ] || { echo "Missing SMD HDF5: $HDF5_VIEW/smd_sr22050" >&2; exit 1; }

MODEL_TYPE=${MODEL_TYPE:-hpt}
SCORE_METHOD=${SCORE_METHOD:-note_editor}
INPUT2=${INPUT2:-onset}
INPUT3=${INPUT3:-frame}
LOSS_TYPE=${LOSS_TYPE:-kim_bce_l1}
PROXY_WEIGHT=${PROXY_WEIGHT:-1.0}
DDSP_PHASE=${DDSP_PHASE:-1}
DDSP_CKPT_EPOCH=${DDSP_CKPT_EPOCH:-7}
LOGRMS_WEIGHT=${LOGRMS_WEIGHT:-0.05}
DDSP_LOUDNESS_WEIGHT=${DDSP_LOUDNESS_WEIGHT:-0.05}

SEGMENTS=("2" "5")
AUDIO_LOSSES=(
  "piano_ssm_spectral"
  "piano_ssm_spectral_plus_log_rms"
  "piano_ssm_spectral_plus_ddsp_loudness"
  "piano_ssm_combined_rm"
)

EXP_NAME=()
EXP_SEGMENT=()
EXP_AUDIO_LOSS=()
EXP_DDSP_CKPT=()

for SEGMENT_SECONDS in "${SEGMENTS[@]}"; do
  DDSP_CKPT="../kaya_data/ddsp-piano-pytorch/workspaces_unified_${SEGMENT_SECONDS}s/models/phase_${DDSP_PHASE}/ckpts/ddsp-piano_epoch_${DDSP_CKPT_EPOCH}_params.pt"
  for AUDIO_LOSS in "${AUDIO_LOSSES[@]}"; do
    EXP_NAME+=("route3_ddsp_${SEGMENT_SECONDS}s_${AUDIO_LOSS}")
    EXP_SEGMENT+=("$SEGMENT_SECONDS")
    EXP_AUDIO_LOSS+=("$AUDIO_LOSS")
    EXP_DDSP_CKPT+=("$DDSP_CKPT")
  done
done

TOTAL_JOBS=${#EXP_NAME[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((TOTAL_JOBS-1)))"
  exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
EXP_TAG=${EXP_NAME[$IDX]}
SEGMENT_SECONDS=${EXP_SEGMENT[$IDX]}
AUDIO_LOSS=${EXP_AUDIO_LOSS[$IDX]}
DDSP_CKPT=${EXP_DDSP_CKPT[$IDX]}

if [ ! -f "$DDSP_CKPT" ]; then
  echo "Missing DDSP-Piano checkpoint: $DDSP_CKPT" >&2
  exit 1
fi

echo "Experiment       : $EXP_TAG"
echo "Model            : $MODEL_TYPE"
echo "Score method     : $SCORE_METHOD"
echo "Input2 / Input3  : $INPUT2 / $INPUT3"
echo "Velocity loss    : $LOSS_TYPE"
echo "Backend seg (s)  : $SEGMENT_SECONDS"
echo "Proxy audio loss : $AUDIO_LOSS"
echo "DDSP checkpoint  : $DDSP_CKPT"

python pytorch/train_ddsp.py \
  exp.workspace="$WORKSPACE_DIR" \
  exp.batch_size=4 \
  dataset.train_set=maestro \
  dataset.test_set=maestro \
  'dataset.eval_sets=[train,maestro,smd]' \
  model.type="$MODEL_TYPE" \
  model.input2="$INPUT2" \
  model.input3="$INPUT3" \
  score_informed.method="$SCORE_METHOD" \
  loss.loss_type="$LOSS_TYPE" \
  loss.supervised_weight=0.0 \
  loss.proxy_weight="$PROXY_WEIGHT" \
  loss.velocity_prior_weight=0.0 \
  proxy.enabled=true \
  proxy.type=diffsynth_piano \
  proxy.project_root=../synthesizer/ddsp-piano-pytorch \
  proxy.checkpoint="$DDSP_CKPT" \
  proxy.backend_segment_seconds="$SEGMENT_SECONDS" \
  proxy.audio_loss.type="$AUDIO_LOSS" \
  proxy.audio_loss.piano_ssm_spectral_plus_log_rms.log_rms_weight="$LOGRMS_WEIGHT" \
  proxy.audio_loss.piano_ssm_spectral_plus_ddsp_loudness.loudness_weight="$DDSP_LOUDNESS_WEIGHT" \
  wandb.comment="$EXP_TAG"

[ -d "$WORKSPACE_DIR/checkpoints" ] && mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"

cd "$HOME"
rm -r "$SCRATCH"
conda deactivate 2>/dev/null || source deactivate 2>/dev/null || deactivate 2>/dev/null || true
echo route3_ablation $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
