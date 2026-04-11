# Kaya Route III / IV

这几个脚本按 `202510_smc/run_scripts` 的老模版写，保留了：

- `cp -r` 到 scratch
- `workspaces/hdf5s` 软链到 scratch 数据根
- DDSP / SFProxy checkpoint 直接走 `$MYSCRATCH/202604_midiproxy_data/...`
- `SLURM_ARRAY_TASK_ID` 展开 ablation 组合
- 训练结束后把 `checkpoints/` 和 `logs/` 挪回 `${MYGROUP}`

脚本：

- `kaya_hpt_route3_ablation.sh`
- `kaya_hpt_route4_ablation.sh`
- `kaya_hpt_route3_ablation_pretrain.sh`
- `kaya_hpt_route4_ablation_pretrain.sh`
- `kaya_hpt_route3_single.sh`
- `kaya_hpt_route4_single.sh`

## Kaya 上要准备的数据

默认假设数据已经放在：

```bash
$MYSCRATCH/202604_midiproxy_data
```

这两个脚本只需要下面这些子树。

### 1. Score-HPT HDF5

必须有：

```text
202604_midiproxy_data/
  score_hpt/
    workspaces/
      hdf5s/
        maestro_sr22050/
          2004/*.h5
          2006/*.h5
          ...
          2018/*.h5
        smd_sr22050/
          *.h5
```

这部分现在需要两套：`maestro_sr22050` 和 `smd_sr22050`。

### 2. Route III 需要的 DDSP-Piano ckpt

必须有：

```text
202604_midiproxy_data/
  ddsp-piano-pytorch/
    workspaces_unified_2s/
      models/phase_1/ckpts/ddsp-piano_epoch_7_params.pt
    workspaces_unified_5s/
      models/phase_1/ckpts/ddsp-piano_epoch_7_params.pt
```

当前本地实际已经看到：

- `2s` phase-1 epoch-1..7
- `5s` phase-1 epoch-1..7

### 3. Route IV 需要的 SFProxy ckpt

必须有：

```text
202604_midiproxy_data/
  synth-proxy/
    proxy/
      checkpoints/
        salamander_piano/
          piano_salamander_piano_coverage_v2_b0_c1_r0_s0_2s_default/*.ckpt
          piano_salamander_piano_coverage_v2_b0_c1_r0_s0_5s_default/*.ckpt
          piano_salamander_piano_coverage_v2_b0_c1_r0_s0_10s_default/*.ckpt
          piano_salamander_piano_mixed_v2_b0p3_c0p4_r0p2_s0p1_2s_default/*.ckpt
          piano_salamander_piano_mixed_v2_b0p3_c0p4_r0p2_s0p1_5s_default/*.ckpt
          piano_salamander_piano_mixed_v2_b0p3_c0p4_r0p2_s0p1_10s_default/*.ckpt
          piano_salamander_piano_realism_v2_b0_c0_r1_s0_2s_default/*.ckpt
          piano_salamander_piano_realism_v2_b0_c0_r1_s0_5s_default/*.ckpt
          piano_salamander_piano_realism_v2_b0_c0_r1_s0_10s_default/*.ckpt
```

脚本默认取：

- `SFPROXY_CKPT_KIND=final`: `*_e199.ckpt`
- `SFPROXY_CKPT_KIND=best`: `*_e*_loss*.ckpt` 里最新那个

## 这次不需要搬的东西

这两份 Kaya 脚本当前不依赖：

- `ddsp-piano-pytorch/workspaces_unified_*/data_cache`
- `synth-proxy/data/...`
- `adv_renders/...`
- `soundfont/...`

也就是说，这次 Route III / IV ablation 只需要：

1. `score_hpt` 的 `maestro + smd` HDF5
2. 冻结的 `ddsp-piano` ckpt
3. 冻结的 `sfproxy` ckpt

## 运行

```bash
sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
```

pretrain continuation：

```bash
HPT_PRETRAINED_CHECKPOINT=/abs/path/to/hpt_checkpoint.pth \
FILMUNET_PRETRAINED_CHECKPOINT=/abs/path/to/filmunet_checkpoint.pth \
sbatch kaya_scripts/kaya_hpt_route3_ablation_pretrain.sh

HPT_PRETRAINED_CHECKPOINT=/abs/path/to/hpt_checkpoint.pth \
FILMUNET_PRETRAINED_CHECKPOINT=/abs/path/to/filmunet_checkpoint.pth \
sbatch kaya_scripts/kaya_hpt_route4_ablation_pretrain.sh
```

当前默认 sweep 维度：

- Route III:
  - `SEGMENTS=("2" "5")`
  - `AUDIO_LOSSES=("piano_ssm_spectral" "piano_ssm_spectral_plus_log_rms" "piano_ssm_spectral_plus_ddsp_loudness" "piano_ssm_combined_rm")`
  - `MODEL_TYPES=("hpt" "filmunet")`
  - `SUP_BACKEND_PAIRS=("0.0,1.0" "0.5,0.5")`
  - `PRIOR_WEIGHTS=("0.0" "0.01")`

- Route IV:
  - `SAMPLERS=("coverage" "mixed" "realism")`
  - `SEGMENTS=("2" "5" "10")`
  - `PROXY_LOSSES=("smooth_l1" "l1" "mse")`
  - `MODEL_TYPES=("hpt" "filmunet")`
  - `SUP_BACKEND_PAIRS=("0.0,1.0" "0.5,0.5")`
  - `PRIOR_WEIGHTS=("0.0" "0.01")`

pretrain wrapper 默认会把 `MODEL_TYPES` 切成：

- `MODEL_TYPES="hpt_pretrained filmunet_pretrained"`

如果你包含某个 `*_pretrained` 模型类型，就需要提供对应 checkpoint：

- `HPT_PRETRAINED_CHECKPOINT=/abs/path/to/hpt_checkpoint.pth`
- `FILMUNET_PRETRAINED_CHECKPOINT=/abs/path/to/filmunet_checkpoint.pth`

如果你想手动删掉一部分 bad options，直接在脚本里注释数组项就行。

interactive debug：

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=32G --time=08:00:00
bash kaya_scripts/kaya_hpt_route3_single.sh
bash kaya_scripts/kaya_hpt_route4_single.sh
```

默认现在 `score_hpt/pytorch/config/config.yaml` 已经切到：

```yaml
exp:
  workspace: "./workspaces"
```

所以 Kaya 脚本都会在 `score_hpt/` 目录下直接使用相对工作区：

- `workspaces/hdf5s -> $MYSCRATCH/202604_midiproxy_data/score_hpt/workspaces/hdf5s`
- DDSP checkpoint -> `$MYSCRATCH/202604_midiproxy_data/ddsp-piano-pytorch/...`
- SFProxy checkpoint -> `$MYSCRATCH/202604_midiproxy_data/synth-proxy/...`

Kaya 脚本现在会在 scratch 副本里先删掉 repo 自带的 `score_hpt/workspaces` 符号链接，再用最直接的方式只建 HDF5 入口：

- `workspaces/hdf5s -> $MYSCRATCH/202604_midiproxy_data/score_hpt/workspaces/hdf5s`

可选：

```bash
SFPROXY_CKPT_KIND=best sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
PROJECT_NAME=202604_midiproxy DATA_PROJECT=202604_midiproxy_data sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
MODEL_TYPES="hpt" SUP_BACKEND_PAIRS="0.5,0.5" PRIOR_WEIGHTS="0.0 0.01" sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
MODEL_TYPES="filmunet" SUP_BACKEND_PAIRS="0.0,1.0 0.5,0.5" PRIOR_WEIGHTS="0.0" sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
MODEL_TYPES="hpt_pretrained" HPT_PRETRAINED_CHECKPOINT=/abs/path/to/hpt_checkpoint.pth sbatch kaya_scripts/kaya_hpt_route3_ablation_pretrain.sh
MODEL_TYPES="filmunet_pretrained" FILMUNET_PRETRAINED_CHECKPOINT=/abs/path/to/filmunet_checkpoint.pth sbatch kaya_scripts/kaya_hpt_route4_ablation_pretrain.sh
SEGMENT_SECONDS=5 AUDIO_LOSS=piano_ssm_spectral bash kaya_scripts/kaya_hpt_route3_single.sh
SAMPLER=realism SEGMENT_SECONDS=5 PROXY_LOSS=l1 bash kaya_scripts/kaya_hpt_route4_single.sh
```

默认不在 Kaya sweep 里打开 BSSL / BSTL。若你之后确实想在某个批次上打开，可以手动传：

```bash
ENABLE_AUDIO_METRICS=1 \
INSTRUMENT_PATH=/abs/path/to/your.sfz \
AUDIO_METRIC_MAX_SEGMENTS=1 \
sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
```
