#!/bin/bash
set -euo pipefail

# Interactive debug helper for one Route IV run on Kaya.
# Example:
#   SAMPLER=mixed SEGMENT_SECONDS=2 PROXY_LOSS=smooth_l1 bash kaya_scripts/kaya_hpt_route4_single.sh

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

SAMPLER=${SAMPLER:-mixed}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-2}
PROXY_LOSS=${PROXY_LOSS:-smooth_l1}
SFPROXY_CKPT_KIND=${SFPROXY_CKPT_KIND:-final}
SFPROXY_FINAL_EPOCH=${SFPROXY_FINAL_EPOCH:-199}

KEEP_SCRATCH=${KEEP_SCRATCH:-1}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-route4_debug_${SAMPLER}_${SEGMENT_SECONDS}s_${PROXY_LOSS}_${RUN_STAMP}}

SCRATCH_PARENT=${SCRATCH_PARENT:-$MYSCRATCH/${PROJECT_NAME}}
RESULTS_PARENT=${RESULTS_PARENT:-$MYGROUP/${PROJECT_NAME}_results}
SCRATCH=${SCRATCH_PARENT}/${RUN_NAME}
RESULTS=${RESULTS_PARENT}/${RUN_NAME}

sampler_dir() {
  local sampler="$1"
  local segment="$2"
  case "$sampler" in
    coverage) echo "piano_salamander_piano_coverage_v2_b0_c1_r0_s0_${segment}s_default" ;;
    mixed) echo "piano_salamander_piano_mixed_v2_b0p3_c0p4_r0p2_s0p1_${segment}s_default" ;;
    realism) echo "piano_salamander_piano_realism_v2_b0_c0_r1_s0_${segment}s_default" ;;
    *) return 1 ;;
  esac
}

resolve_sfproxy_ckpt() {
  local sampler="$1"
  local segment="$2"
  local run_dir
  local base_dir
  local best_ckpt

  run_dir="$(sampler_dir "$sampler" "$segment")" || return 1
  base_dir="$SFPROXY_ROOT/proxy/checkpoints/salamander_piano/${run_dir}"

  case "$SFPROXY_CKPT_KIND" in
    final)
      printf '%s/%s_e%s.ckpt\n' "$base_dir" "$run_dir" "$SFPROXY_FINAL_EPOCH"
      ;;
    best)
      best_ckpt="$(find "$base_dir" -maxdepth 1 -type f -name "${run_dir}_e*_loss*.ckpt" | sort -V | tail -n 1)"
      if [ -n "$best_ckpt" ]; then
        printf '%s\n' "$best_ckpt"
      else
        printf '%s/%s_e%s.ckpt\n' "$base_dir" "$run_dir" "$SFPROXY_FINAL_EPOCH"
      fi
      ;;
    *)
      echo "Unsupported SFPROXY_CKPT_KIND='$SFPROXY_CKPT_KIND'" >&2
      return 1
      ;;
  esac
}

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
SFPROXY_ROOT="$DATA_ROOT/synth-proxy"

ln -s "$HDF5_SRC" "$HDF5_VIEW"

[ -d "$HDF5_VIEW/maestro_sr22050" ] || { echo "Missing MAESTRO HDF5: $HDF5_VIEW/maestro_sr22050" >&2; exit 1; }
[ -d "$HDF5_VIEW/smd_sr22050" ] || { echo "Missing SMD HDF5: $HDF5_VIEW/smd_sr22050" >&2; exit 1; }

PROXY_CKPT="$(resolve_sfproxy_ckpt "$SAMPLER" "$SEGMENT_SECONDS")"
if [ ! -f "$PROXY_CKPT" ]; then
  echo "Missing SFProxy checkpoint: $PROXY_CKPT" >&2
  exit 1
fi

echo "Run name         : $RUN_NAME"
echo "Model            : $MODEL_TYPE"
echo "Score method     : $SCORE_METHOD"
echo "Input2 / Input3  : $INPUT2 / $INPUT3"
echo "Velocity loss    : $LOSS_TYPE"
echo "Sampler          : $SAMPLER"
echo "Backend seg (s)  : $SEGMENT_SECONDS"
echo "Proxy loss       : $PROXY_LOSS"
echo "SFProxy ckpt     : $PROXY_CKPT"

python pytorch/train_proxy.py \
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
  proxy.type=diffproxy \
  proxy.project_root=../synth-proxy \
  proxy.checkpoint="$PROXY_CKPT" \
  proxy.backend_segment_seconds="$SEGMENT_SECONDS" \
  proxy.warmup_iterations=0 \
  proxy.supervision.hop_size=221 \
  proxy.sfproxy.instrument_name=salamander_piano \
  proxy.sfproxy.loss_type="$PROXY_LOSS" \
  proxy.sfproxy.use_gt_aligned_note_events=true \
  proxy.sfproxy.feature.hop=221 \
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
