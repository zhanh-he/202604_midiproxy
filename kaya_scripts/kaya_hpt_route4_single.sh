#!/bin/bash
#SBATCH --job-name=route4_single
#SBATCH --output=route4_single_progress_%j.log
#SBATCH --error=route4_single_error_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

# Single-job debug version of route4 ablation.
# Example:
#   sbatch --export=ALL,SEGMENT_SECONDS=2 kaya_hpt_route4_single.sh

module load Anaconda3/2024.06 cuda/12.4.1 gcc/12.4.0
module list
source activate bark_env

echo "Running on host: $(hostname)"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM JOB ID: ${SLURM_JOB_ID:-N/A}"

FOLDER_NAME=${SLURM_JOB_ID:-route4_single}
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
HDF5_SRC="$DATA_ROOT/score_hpt/workspaces/hdf5s"
HDF5_VIEW="$WORKSPACE_DIR/hdf5s"
SFPROXY_ROOT="$DATA_ROOT/synth-proxy"

ln -s "$HDF5_SRC" "$HDF5_VIEW"

[ -d "$HDF5_VIEW/maestro_sr22050" ] || { echo "Missing MAESTRO HDF5: $HDF5_VIEW/maestro_sr22050" >&2; exit 1; }
[ -d "$HDF5_VIEW/smd_sr22050" ] || { echo "Missing SMD HDF5: $HDF5_VIEW/smd_sr22050" >&2; exit 1; }

SEGMENT_SECONDS=${SEGMENT_SECONDS:-2}
MODEL_TYPE=${MODEL_TYPE:-hpt}
SCORE_METHOD=${SCORE_METHOD:-note_editor}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
LOSS_TYPE=${LOSS_TYPE:-kim_bce_l1}
SUPERVISED_WEIGHT=${SUPERVISED_WEIGHT:-0.0}
BACKEND_WEIGHT=${BACKEND_WEIGHT:-1.0}
PRIOR_WEIGHT=${PRIOR_WEIGHT:-0.0}
SAMPLER=${SAMPLER:-mixed}
PROXY_LOSS=${PROXY_LOSS:-smooth_l1}
SFPROXY_CKPT_KIND=${SFPROXY_CKPT_KIND:-final}
SFPROXY_FINAL_EPOCH=${SFPROXY_FINAL_EPOCH:-199}
ENABLE_AUDIO_METRICS=${ENABLE_AUDIO_METRICS:-0}
INSTRUMENT_PATH=${INSTRUMENT_PATH:-}
AUDIO_METRIC_MAX_SEGMENTS=${AUDIO_METRIC_MAX_SEGMENTS:-4}

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

PROXY_CKPT="$(resolve_sfproxy_ckpt "$SAMPLER" "$SEGMENT_SECONDS")"

RUN_SCORE_METHOD="$SCORE_METHOD"
if [ "$MODEL_TYPE" != "hpt" ] && [ "$MODEL_TYPE" != "filmunet" ]; then
  echo "Unsupported MODEL_TYPE: $MODEL_TYPE. Use hpt or filmunet." >&2
  exit 1
fi
if [ "$MODEL_TYPE" = "filmunet" ]; then
  RUN_SCORE_METHOD="direct"
elif [ "$RUN_SCORE_METHOD" != "direct" ] && [ "$RUN_SCORE_METHOD" != "note_editor" ]; then
  echo "Unsupported SCORE_METHOD for $MODEL_TYPE: $RUN_SCORE_METHOD" >&2
  exit 1
fi
MODEL_INPUT2="null"
if [ "$MODEL_TYPE" = "hpt" ] && [ "$RUN_SCORE_METHOD" = "note_editor" ]; then
  MODEL_INPUT2="onset"
fi

if [ ! -f "$PROXY_CKPT" ]; then
  echo "Missing SFProxy checkpoint: $PROXY_CKPT" >&2
  exit 1
fi
if [ -n "$PRETRAINED_CHECKPOINT" ] && [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
  echo "Pretrained checkpoint not found: $PRETRAINED_CHECKPOINT" >&2
  exit 1
fi

sup_tag=${SUPERVISED_WEIGHT/./p}
backend_tag=${BACKEND_WEIGHT/./p}
prior_tag=${PRIOR_WEIGHT/./p}
EXP_TAG=${EXP_TAG:-route4_${SAMPLER}_${SEGMENT_SECONDS}s_${PROXY_LOSS}_${MODEL_TYPE}_sup${sup_tag}_backend${backend_tag}_prior${prior_tag}}

echo "Experiment       : $EXP_TAG"
echo "Model            : $MODEL_TYPE"
echo "Score method     : $RUN_SCORE_METHOD"
echo "Input2 / Input3  : $MODEL_INPUT2 / null"
echo "Velocity loss    : $LOSS_TYPE"
echo "Supervised weight: $SUPERVISED_WEIGHT"
echo "Backend weight   : $BACKEND_WEIGHT"
echo "Prior weight     : $PRIOR_WEIGHT"
echo "Sampler          : $SAMPLER"
echo "Backend seg (s)  : $SEGMENT_SECONDS"
echo "Backend loss     : $PROXY_LOSS"
echo "SFProxy ckpt     : $PROXY_CKPT"

EXTRA_ARGS=()
if [ "$ENABLE_AUDIO_METRICS" = "1" ] || [ "$ENABLE_AUDIO_METRICS" = "true" ]; then
  if [ -z "$INSTRUMENT_PATH" ]; then
    echo "INSTRUMENT_PATH must be set when ENABLE_AUDIO_METRICS=1" >&2
    exit 1
  fi
  EXTRA_ARGS+=(
    "train_eval.audio_metrics.enabled=true"
    "train_eval.audio_metrics.instrument_path=$INSTRUMENT_PATH"
    "train_eval.audio_metrics.max_segments=$AUDIO_METRIC_MAX_SEGMENTS"
  )
fi
if [ -n "$PRETRAINED_CHECKPOINT" ]; then
  EXTRA_ARGS+=("model.pretrained_checkpoint=$PRETRAINED_CHECKPOINT")
fi

python pytorch/train_proxy.py \
  exp.workspace="$WORKSPACE_DIR" \
  dataset.train_set=maestro \
  dataset.test_set=maestro \
  'dataset.eval_sets=[train,maestro,smd]' \
  model.type="$MODEL_TYPE" \
  model.input2="$MODEL_INPUT2" \
  model.input3=null \
  score_informed.method="$RUN_SCORE_METHOD" \
  loss.loss_type="$LOSS_TYPE" \
  loss.supervised_weight="$SUPERVISED_WEIGHT" \
  loss.proxy_weight="$BACKEND_WEIGHT" \
  loss.velocity_prior_weight="$PRIOR_WEIGHT" \
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
  wandb.comment="$EXP_TAG" \
  "${EXTRA_ARGS[@]}"

[ -d "$WORKSPACE_DIR/checkpoints" ] && mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"

cd "$HOME"
rm -r "$SCRATCH"
conda deactivate 2>/dev/null || source deactivate 2>/dev/null || deactivate 2>/dev/null || true
echo route4_single ${SLURM_JOB_ID:-N/A} finished at "$(date)"
