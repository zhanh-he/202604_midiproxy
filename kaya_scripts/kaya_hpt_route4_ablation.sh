#!/bin/bash
#SBATCH --job-name=route4_ablation
#SBATCH --output=route4_ablation_progress_%A_%a.log
#SBATCH --error=route4_ablation_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-26
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
SFPROXY_CKPT_KIND=${SFPROXY_CKPT_KIND:-final}
SFPROXY_FINAL_EPOCH=${SFPROXY_FINAL_EPOCH:-199}

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
  base_dir="../kaya_data/synth-proxy/proxy/checkpoints/salamander_piano/${run_dir}"

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

SAMPLERS=("coverage" "mixed" "realism")
SEGMENTS=("2" "5" "10")
PROXY_LOSSES=("smooth_l1" "l1" "mse")

EXP_NAME=()
EXP_SAMPLER=()
EXP_SEGMENT=()
EXP_PROXY_LOSS=()
EXP_PROXY_CKPT=()

for SAMPLER in "${SAMPLERS[@]}"; do
  for SEGMENT_SECONDS in "${SEGMENTS[@]}"; do
    PROXY_CKPT="$(resolve_sfproxy_ckpt "$SAMPLER" "$SEGMENT_SECONDS")"
    for PROXY_LOSS in "${PROXY_LOSSES[@]}"; do
      EXP_NAME+=("route4_${SAMPLER}_${SEGMENT_SECONDS}s_${PROXY_LOSS}")
      EXP_SAMPLER+=("$SAMPLER")
      EXP_SEGMENT+=("$SEGMENT_SECONDS")
      EXP_PROXY_LOSS+=("$PROXY_LOSS")
      EXP_PROXY_CKPT+=("$PROXY_CKPT")
    done
  done
done

TOTAL_JOBS=${#EXP_NAME[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((TOTAL_JOBS-1)))"
  exit 1
fi

IDX=$SLURM_ARRAY_TASK_ID
EXP_TAG=${EXP_NAME[$IDX]}
SAMPLER=${EXP_SAMPLER[$IDX]}
SEGMENT_SECONDS=${EXP_SEGMENT[$IDX]}
PROXY_LOSS=${EXP_PROXY_LOSS[$IDX]}
PROXY_CKPT=${EXP_PROXY_CKPT[$IDX]}

if [ ! -f "$PROXY_CKPT" ]; then
  echo "Missing SFProxy checkpoint: $PROXY_CKPT" >&2
  exit 1
fi

echo "Experiment       : $EXP_TAG"
echo "Model            : $MODEL_TYPE"
echo "Score method     : $SCORE_METHOD"
echo "Input2 / Input3  : $INPUT2 / $INPUT3"
echo "Velocity loss    : $LOSS_TYPE"
echo "Sampler          : $SAMPLER"
echo "Backend seg (s)  : $SEGMENT_SECONDS"
echo "Proxy loss       : $PROXY_LOSS"
echo "SFProxy ckpt     : $PROXY_CKPT"

python pytorch/train_proxy.py \
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
  wandb.comment="$EXP_TAG"

[ -d "$WORKSPACE_DIR/checkpoints" ] && mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
[ -d "$WORKSPACE_DIR/logs" ] && mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"

cd "$HOME"
rm -r "$SCRATCH"
conda deactivate 2>/dev/null || source deactivate 2>/dev/null || deactivate 2>/dev/null || true
echo route4_ablation $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
