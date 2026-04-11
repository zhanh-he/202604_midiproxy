# Score-HPT + SFProxy Notes

这份 README 只保留当前项目里已经打通、还在使用、并且适合继续做 loss ablation 的内容。

当前 repo 已经比较成熟，可以把它理解成两条主线：

- `synth-proxy/`
  - 准备 SoundFont teacher data
  - 训练 SFProxy
  - 做 preset / segment / instrument ablation

- `score_hpt/pytorch/`
  - 训练 velocity model
  - Route II: 纯监督
  - Route III: 接 DiffSynth / DDSP backend loss
  - Route IV: 接 frozen SFProxy backend loss

## 1. 现在的训练入口

常用入口：

- [`pytorch/train.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/train.py)
  - Route II
  - 纯监督 velocity estimation

- [`pytorch/train_ddsp.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/train_ddsp.py)
  - Route III
  - `proxy.type=diffsynth_piano`
  - 默认 `loss.proxy_weight=1.0`

- [`pytorch/train_proxy.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/train_proxy.py)
  - Route IV
  - `proxy.type=diffproxy`
  - 默认 `loss.proxy_weight=1.0`

最常见的 Route IV 形式：

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt

python pytorch/train_proxy.py \
  proxy.checkpoint=/path/to/sfproxy.ckpt \
  proxy.sfproxy.instrument_name=salamander_piano
```

## 2. 支持的 base model

当前训练链已经支持：

- `hpt`
- `hpt_pretrained`
- `hppnet`
- `dynest`
- `filmunet`
- `filmunet_pretrained`

其中：

- `hpt` / `filmunet` 是从零训练
- `hpt_pretrained` / `filmunet_pretrained` 是继续训练，要求提供 `model.pretrained_checkpoint`

默认建议：

- `model.kim_condition=frame`

这在当前 repo 里已经是默认值，不需要额外再写。

如果要从零训练 FiLMUNet：

```bash
python pytorch/train.py \
  model.type=filmunet
```

如果要继续训练 pretrained FiLMUNet：

```bash
python pytorch/train.py \
  model.type=filmunet_pretrained \
  model.pretrained_checkpoint=/path/to/FiLMUNetPretrained+frame/1000000_iterations.pth
```

Route III / IV 同理，只需要把入口改成 `train_ddsp.py` 或 `train_proxy.py`。

## 3. Loss 总览

### Route II: supervised velocity loss

在 [`losses.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py) 里当前可用的是：

- `velocity_bce`
- `velocity_mse`
- `kim_bce_l1`

推荐默认：

- `kim_bce_l1`

原因：

- 它已经是当前 repo 的默认配置
- 对齐了我们现在的 route2 / score-informed 主线
- 比单独 `velocity_bce` 或 `velocity_mse` 更平衡

### Route III: differentiable audio proxy loss

在 [`config.yaml`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/config/config.yaml) 里当前可选的是：

- `piano_ssm_combined`
- `piano_ssm_combined_rm`
- `piano_ssm_spectral`
- `piano_ssm_spectral_plus_log_rms`
- `piano_ssm_spectral_plus_ddsp_loudness`
- `ddsp_piano_loudness`

推荐默认：

- `piano_ssm_spectral_plus_log_rms`

原因：

- 主体是 Piano-SSM spectral loss，比较稳
- 附加一个小权重的 log-RMS loudness bias
- 比纯 spectral 更容易观察 dynamics
- 比 frame-wise loudness contour 更保守，适合作为 Route III 基线

更强的备选：

- `piano_ssm_spectral_plus_ddsp_loudness`

它更强调 frame-wise loudness contour，更 aggressive，适合后续 ablation。

### Route IV: frozen SFProxy loss

当前 `proxy.sfproxy.loss_type` 支持：

- `smooth_l1`
- `l1`
- `mse`

推荐默认：

- `smooth_l1`

原因：

- Route IV 的 target 不是直接 GT velocity，而是 proxy teacher 提取出的 note-level dynamics feature
- `smooth_l1` 对噪声和异常点更稳
- `mse` 通常太容易被大误差点主导
- `l1` 可以作为对照，但一般不建议优先于 `smooth_l1`

推荐排序：

1. `smooth_l1`
2. `l1`
3. `mse`

### 通用辅助项

- `velocity_prior_loss`

这是附加 regularizer，不是主 loss。

## 4. 现在推荐怎么做 loss ablation

如果目标是“先把一条清晰主线跑稳，再做扩展”，建议：

### Route II

先固定：

- `loss.loss_type=kim_bce_l1`

base model 可以扫：

- `hpt`
- `hpt_pretrained`
- `hppnet`
- `dynest`
- `filmunet`
- `filmunet_pretrained`

### Route III

先固定：

- `proxy.type=diffsynth_piano`
- `proxy.audio_loss.type=piano_ssm_spectral_plus_log_rms`

然后 ablate：

- `piano_ssm_spectral`
- `piano_ssm_spectral_plus_log_rms`
- `piano_ssm_spectral_plus_ddsp_loudness`
- `piano_ssm_combined_rm`

如果想把 loudness 这一轴看清楚，可以优先比较：

- `piano_ssm_spectral`
- `piano_ssm_spectral_plus_log_rms`
- `piano_ssm_spectral_plus_ddsp_loudness`

### Route IV

先固定：

- `proxy.sfproxy.loss_type=smooth_l1`

然后做最直接的 teacher-loss ablation：

- `smooth_l1`
- `l1`
- `mse`

### 一个实用的顺序

建议顺序：

1. 固定一个 base model
2. 先在 Route III 扫 audio loss
3. 再在 Route IV 扫 `smooth_l1 / l1 / mse`
4. 最后才换 base model 或 score-informed method

这样变量更干净。

## 5. Route III / IV 当前 wandb 会记录什么

### Charts 里会看到的

训练侧常见指标：

- `train_total_loss`
- `train_proxy_loss`
- `train_proxy_*`

其中 Route III 默认 `piano_ssm_spectral_plus_log_rms` 会额外带出：

- `train_proxy_spectral_raw`
- `train_proxy_spectral_weighted`
- `train_proxy_log_rms_raw`
- `train_proxy_log_rms_weighted`
- `train_proxy_spectral_mag_*`
- `train_proxy_spectral_logmag_*`
- `train_proxy_log_rms_log_rms_loss`
- `train_proxy_log_rms_log_rms_pred`
- `train_proxy_log_rms_log_rms_target`
- `train_proxy_*audio_rms_*`

验证侧常见指标：

- `train_stat.frame_max_error`
- `train_stat.frame_max_std`
- `train_stat.onset_masked_error`
- `train_stat.onset_masked_std`
- `valid_maestro_stat.*`
- `valid_smd_stat.*`

如果手动打开 `train_eval.audio_metrics.enabled=true`，还会多出：

- `real_pred_bssl_pearson_correlation`
- `real_pred_bstl_pearson_correlation`

### 图像类

现在 Route II / III / IV 都会在 wandb 里给：

- `velocity_roll_comparison`

这张图对肉眼理解 velocity learning 很有用。

## 6. 哪些常数已经从 wandb charts 剥离

下面这些常数型 renderer / supervision 配置不再进 charts，只写到 summary 和本地 log：

- `proxy_loss_sample_rate`
- `proxy_loss_frame_rate`
- `proxy_proxy_frames`
- `proxy_proxy_polyphony`
- `proxy_renderer_sample_rate`
- `proxy_renderer_frame_rate`
- `proxy_renderer_segment_seconds`
- `proxy_synth_input_source_fps`
- `proxy_renderer_hop_length`
- `proxy_renderer_n_fft`
- `proxy_native_segment_seconds`

## 7. 训练日志现在怎么落盘

Route III / IV 在每个 checkpoint 目录下都会追加：

- `training_stats.txt`
- `training.log`

其中 `training.log` 会记录每次 eval / checkpoint 时的：

- `train:`
- `eval/train:`
- `eval/maestro:`
- `eval/smd:`
- `summary:`

这个文件是后续和 Codex 讨论结果时最方便的文本入口，因为 wandb 页面内容我看不到，但你可以直接贴 log。

## 8. 可选 audio Pearson eval

当前训练已经支持可选的 render-audio Pearson eval。

配置在：

- `train_eval.audio_metrics.enabled`

默认：

- `false`

原因：

- 在集群和部分 3090 机器上不稳定
- 需要本地可用的 SF2 / SFZ 渲染环境

适合开启的场景：

- 5090 本地机器
- 已确认 `instrument_path` 可用
- 想观察 “预测 velocity -> render audio -> real audio Pearson” 这条指标

示例：

```bash
python pytorch/train_ddsp.py \
  train_eval.audio_metrics.enabled=true \
  train_eval.audio_metrics.instrument_path=/path/to/your.sfz
```

## 9. SFProxy 数据准备

如果你现在要先把 SFProxy 路线跑通，优先看：

- [preprocess_sfproxy_data.sh](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/preprocess_sfproxy_data.sh)
- [train_sfproxy.sh](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_sfproxy.sh)
- [train_sfproxy_ablations.sh](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_sfproxy_ablations.sh)
- [eval.ipynb](/media/mengh/SharedData/zhanh/202604_midiproxy/synth-proxy/eval.ipynb)

默认数据准备：

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts
bash preprocess_sfproxy_data.sh
```

默认含义：

- `INSTRUMENT=piano`
- `PIANO_DATASET=maestro`
- 输出乐器名 `salamander_piano`
- `BOUNDARY_MODE=default`

脚本会先导出：

- `boundary_v2`
- `coverage_v2`
- `realism_v2`
- `stress_v2`

然后 compose 出：

- `mixed_v2`

常用改法：

```bash
SEGMENT_SECONDS=2 bash preprocess_sfproxy_data.sh
SEGMENT_SECONDS=5 bash preprocess_sfproxy_data.sh
SEGMENT_SECONDS=10 bash preprocess_sfproxy_data.sh
```

如果 component 数据已经存在，重新 compose `mixed_v2` 会明显比重渲染快。

## 10. SFProxy 训练主线

单个模型：

```bash
bash train_sfproxy.sh
```

默认：

- `TRAIN_PRESET=mixed_v2`
- `SEGMENT_SECONDS=2`
- `BOUNDARY_MODE=default`

常见 piano 主线：

- `coverage_v2`, `2s`, `default`
- `realism_v2`, `2s`, `default`
- `stress_v2`, `2s`, `default`
- `mixed_v2`, `2s`, `default`

后续再补：

- `mixed_v2`, `5s`, `default`
- `mixed_v2`, `10s`, `default`

批量 ablation：

```bash
bash train_sfproxy_ablations.sh
```

补 `5s / 10s`：

```bash
SEGMENT_LIST="5 10" bash train_sfproxy_ablations.sh
```

一次扫完 `2 / 5 / 10`：

```bash
SEGMENT_LIST="2 5 10" bash train_sfproxy_ablations.sh
```

## 11. 现在已经淘汰的旧说法

下面这些说法现在不要再参考：

- `PREPROCESSING_LAYOUT=monolithic/modular`
- `TARGET_PRESET=v2_bundle`
- 把 `mixed_v2` 理解成在线 sampler 直接生成的唯一主数据

当前真实流程是：

1. 先准备 `boundary / coverage / realism / stress`
2. 再 compose `mixed_v2`
3. 训练时从已有数据目录读取
