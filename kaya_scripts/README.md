# Kaya Route III / IV

这几个脚本按 `202510_smc/run_scripts` 的老模版写，保留了：

- `cp -r` 到 scratch
- `workspaces` 下软链 `hdf5s`
- 显式链接 `hdf5s`、`ddsp-piano-pytorch`、`synth-proxy`
- `SLURM_ARRAY_TASK_ID` 展开 ablation 组合
- 训练结束后把 `checkpoints/` 和 `logs/` 挪回 `${MYGROUP}`

脚本：

- `kaya_hpt_route3_ablation.sh`
- `kaya_hpt_route4_ablation.sh`
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

所以 Kaya 脚本都会在 `score_hpt/` 目录下直接使用相对工作区，并显式建立这三个链接：

- `workspaces/hdf5s -> $MYSCRATCH/202604_midiproxy_data/score_hpt/workspaces/hdf5s`
- `../kaya_data/ddsp-piano-pytorch -> $MYSCRATCH/202604_midiproxy_data/ddsp-piano-pytorch`
- `../kaya_data/synth-proxy -> $MYSCRATCH/202604_midiproxy_data/synth-proxy`

Kaya 脚本现在会在 scratch 副本里先删掉 repo 自带的 `score_hpt/workspaces` 符号链接，再用最直接的方式建三个链接：

- `workspaces/hdf5s -> $MYSCRATCH/202604_midiproxy_data/score_hpt/workspaces/hdf5s`
- `../kaya_data/ddsp-piano-pytorch -> $MYSCRATCH/202604_midiproxy_data/ddsp-piano-pytorch`
- `../kaya_data/synth-proxy -> $MYSCRATCH/202604_midiproxy_data/synth-proxy`

可选：

```bash
SFPROXY_CKPT_KIND=best sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
PROJECT_NAME=202604_midiproxy DATA_PROJECT=202604_midiproxy_data sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
SEGMENT_SECONDS=5 AUDIO_LOSS=piano_ssm_spectral bash kaya_scripts/kaya_hpt_route3_single.sh
SAMPLER=realism SEGMENT_SECONDS=5 PROXY_LOSS=l1 bash kaya_scripts/kaya_hpt_route4_single.sh
```
