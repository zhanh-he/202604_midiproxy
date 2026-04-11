#!/bin/bash
set -euo pipefail

# Interactive debug helper for one Route III run on Kaya.
# Example:
#   SEGMENT_SECONDS=2 AUDIO_LOSS=piano_ssm_spectral_plus_log_rms bash kaya_scripts/kaya_hpt_route3_single.sh

module load Anaconda3/2024.06 cuda/12.4.1 gcc/12.4.0
module list
source activate bark_env

echo "Running on host: $(hostname)"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM JOB ID: ${SLURM_JOB_ID:-N/A}"

PROJECT_NAME=${PROJECT_NAME:-202604_midiproxy}
DATA_PROJECT=${DATA_PROJECT:-202604_midiproxy_data}
EXECUTABLE=${EXECUTABLE:-$HOME/${PROJECT_NAME}}
DATA_ROOT=${DATA_ROOT:-$MYSCRATCH/${DATA_PROJECT}}

MODEL_TYPE=${MODEL_TYPE:-hpt}
SCORE_METHOD=${SCORE_METHOD:-note_editor}
INPUT2=${INPUT2:-onset}
INPUT3=${INPUT3:-frame}
LOSS_TYPE=${LOSS_TYPE:-kim_bce_l1}
PROXY_WEIGHT=${PROXY_WEIGHT:-1.0}

SEGMENT_SECONDS=${SEGMENT_SECONDS:-2}
AUDIO_LOSS=${AUDIO_LOSS:-piano_ssm_spectral_plus_log_rms}
DDSP_PHASE=${DDSP_PHASE:-1}
DDSP_CKPT_EPOCH=${DDSP_CKPT_EPOCH:-7}
LOGRMS_WEIGHT=${LOGRMS_WEIGHT:-0.05}
DDSP_LOUDNESS_WEIGHT=${DDSP_LOUDNESS_WEIGHT:-0.05}

KEEP_SCRATCH=${KEEP_SCRATCH:-1}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-route3_debug_${SEGMENT_SECONDS}s_${AUDIO_LOSS}_${RUN_STAMP}}

SCRATCH_PARENT=${SCRATCH_PARENT:-$MYSCRATCH/${PROJECT_NAME}}
RESULTS_PARENT=${RESULTS_PARENT:-$MYGROUP/${PROJECT_NAME}_results}
SCRATCH=${SCRATCH_PARENT}/${RUN_NAME}
RESULTS=${RESULTS_PARENT}/${RUN_NAME}

mkdir -p "$SCRATCH" "$RESULTS"
echo "SCRATCH is $SCRATCH"
echo "RESULTS dir is $RESULTS"

echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r "$EXECUTABLE" "$SCRATCH"
cd "$SCRATCH/$PROJECT_NAME/score_hpt"

WORKSPACE_DIR=./workspaces
rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"

HDF5_SRC="$DATA_ROOT/score_hpt/workspaces/hdf5s"
HDF5_VIEW="$WORKSPACE_DIR/hdf5s"
DDSP_ROOT="$DATA_ROOT/ddsp-piano-pytorch"

ln -s "$HDF5_SRC" "$HDF5_VIEW"

[ -d "$HDF5_VIEW/maestro_sr22050" ] || { echo "Missing MAESTRO HDF5: $HDF5_VIEW/maestro_sr22050" >&2; exit 1; }
[ -d "$HDF5_VIEW/smd_sr22050" ] || { echo "Missing SMD HDF5: $HDF5_VIEW/smd_sr22050" >&2; exit 1; }

DDSP_CKPT="$DDSP_ROOT/workspaces_unified_${SEGMENT_SECONDS}s/models/phase_${DDSP_PHASE}/ckpts/ddsp-piano_epoch_${DDSP_CKPT_EPOCH}_params.pt"
if [ ! -f "$DDSP_CKPT" ]; then
  echo "Missing DDSP-Piano checkpoint: $DDSP_CKPT" >&2
  exit 1
fi

echo "Run name         : $RUN_NAME"
echo "Model            : $MODEL_TYPE"
echo "Score method     : $SCORE_METHOD"
echo "Input2 / Input3  : $INPUT2 / $INPUT3"
echo "Velocity loss    : $LOSS_TYPE"
echo "Backend seg (s)  : $SEGMENT_SECONDS"
echo "Proxy audio loss : $AUDIO_LOSS"
echo "DDSP checkpoint  : $DDSP_CKPT"

python pytorch/train_ddsp.py \
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
  wandb.comment="$RUN_NAME"

[ -d "$WORKSPACE_DIR/checkpoints" ] && cp -r "$WORKSPACE_DIR/checkpoints" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && cp -r "$WORKSPACE_DIR/logs" "${RESULTS}/"

echo "Finished. Scratch kept at: $SCRATCH"
echo "Results copied to: $RESULTS"

if [ "$KEEP_SCRATCH" = "0" ]; then
  cd "$HOME"
  rm -rf "$SCRATCH"
  echo "Scratch removed."
fi
