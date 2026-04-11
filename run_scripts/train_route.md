# Route III / IV Training Guide

This note is for the current `route3` and `route4` training scripts in this repo.

- `route3` means differentiable audio-backend training via `pytorch/train_ddsp.py`
- `route4` means differentiable note-backend training via `pytorch/train_proxy.py`
- default base model in the ablation scripts is `hpt`
- scratch training model choices are `hpt` and `filmunet`
- pretrained continuation model choices are `hpt_pretrained` and `filmunet_pretrained`
- BSSL / BSTL audio evaluation is off by default
- the default `instrument_path` stays empty on purpose, so Kaya or any machine without local soundfonts will not fail unless you explicitly enable audio evaluation
- route3 / route4 run names now automatically include `sup*` and `backend*` weights

## Local Soundfont Paths

Current local soundfont root:

```text
/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont
```

Recommended local `.sfz` paths:

- piano: `/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz`
- guitar: `/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SpanishClassicalGuitar/SpanishClassicalGuitar-20190618.sfz`

You can also use the `.sf2` versions in the same folders, but `.sfz` is the most natural choice for the current local rendering setup.

## Quick Start

Route III ablation:

```bash
TRAIN_SET=maestro bash run_scripts/train_route3_ablation.sh
TRAIN_SET=francoisleduc bash run_scripts/train_route3_ablation.sh
```

Route IV ablation:

```bash
TRAIN_SET=maestro bash run_scripts/train_route4_ablation.sh
TRAIN_SET=gaps bash run_scripts/train_route4_ablation.sh
```

These two ablation scripts will loop over multiple settings automatically.

- [`train_route3_ablation.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_route3_ablation.sh) sweeps `SEGMENT_LIST` and `LOSS_TYPES`
- [`train_route4_ablation.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_route4_ablation.sh) sweeps `SEGMENT_LIST`, `SAMPLERS`, and `LOSS_TYPES`

## Ablation Commands

### Route III

Default sweep:

```bash
TRAIN_SET=maestro bash run_scripts/train_route3_ablation.sh
```

By default this sweeps:

- `SEGMENT_LIST="2 5"`
- `LOSS_TYPES="piano_ssm_spectral piano_ssm_spectral_plus_log_rms piano_ssm_spectral_plus_ddsp_loudness piano_ssm_combined_rm"`

Useful examples:

```bash
TRAIN_SET=maestro SEGMENT_LIST="2" LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
  bash run_scripts/train_route3_ablation.sh
```

```bash
TRAIN_SET=francoisleduc SEGMENT_LIST="2 5" LOSS_TYPES="piano_ssm_spectral piano_ssm_combined_rm" \
  bash run_scripts/train_route3_ablation.sh
```

Use scratch FiLM-UNet instead of HPT:

```bash
TRAIN_SET=maestro MODEL_TYPE=filmunet SEGMENT_LIST="2" LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
  bash run_scripts/train_route3_ablation.sh
```

Enable optional audio Pearson evaluation:

```bash
TRAIN_SET=maestro SEGMENT_LIST="2" LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
  EXTRA_OVERRIDES="train_eval.audio_metrics.enabled=true train_eval.audio_metrics.instrument_path=/path/to/instrument.sfz" \
  bash run_scripts/train_route3_ablation.sh
```

### Route IV

Default sweep:

```bash
TRAIN_SET=maestro bash run_scripts/train_route4_ablation.sh
```

By default this sweeps:

- piano: `SEGMENT_LIST="2 5 10"`
- guitar: `SEGMENT_LIST="2 5"`
- `SAMPLERS="coverage mixed realism"`
- `LOSS_TYPES="smooth_l1 l1 mse"`

Useful examples:

```bash
TRAIN_SET=maestro SEGMENT_LIST="2" SAMPLERS="mixed" LOSS_TYPES="smooth_l1" \
  bash run_scripts/train_route4_ablation.sh
```

```bash
TRAIN_SET=gaps SEGMENT_LIST="2 5" SAMPLERS="realism" LOSS_TYPES="smooth_l1 l1" \
  bash run_scripts/train_route4_ablation.sh
```

Use scratch FiLM-UNet instead of HPT:

```bash
TRAIN_SET=maestro MODEL_TYPE=filmunet SEGMENT_LIST="2" SAMPLERS="mixed" LOSS_TYPES="smooth_l1" \
  bash run_scripts/train_route4_ablation.sh
```

Use one manually chosen SFProxy checkpoint:

```bash
TRAIN_SET=maestro SEGMENT_LIST="2" LOSS_TYPES="smooth_l1" \
  bash run_scripts/train_route4_ablation.sh /path/to/your_proxy.ckpt
```

## Single-Run Commands

These are useful when you want one controlled run instead of a full sweep.

### Kaya single Route III run

[`kaya_hpt_route3_single.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/kaya_scripts/kaya_hpt_route3_single.sh)

```bash
SEGMENT_SECONDS=2 AUDIO_LOSS=piano_ssm_spectral_plus_log_rms \
  bash kaya_scripts/kaya_hpt_route3_single.sh
```

Change the model:

```bash
MODEL_TYPE=filmunet SEGMENT_SECONDS=2 AUDIO_LOSS=piano_ssm_spectral_plus_log_rms \
  bash kaya_scripts/kaya_hpt_route3_single.sh
```

### Kaya single Route IV run

[`kaya_hpt_route4_single.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/kaya_scripts/kaya_hpt_route4_single.sh)

```bash
SAMPLER=mixed SEGMENT_SECONDS=2 PROXY_LOSS=smooth_l1 \
  bash kaya_scripts/kaya_hpt_route4_single.sh
```

Change the model:

```bash
MODEL_TYPE=filmunet SAMPLER=mixed SEGMENT_SECONDS=2 PROXY_LOSS=smooth_l1 \
  bash kaya_scripts/kaya_hpt_route4_single.sh
```

## Important Environment Variables

Common knobs:

- `TRAIN_SET`: `maestro`, `smd`, `francoisleduc`, `gaps`
- `MODEL_TYPE`: `hpt`, `filmunet`, `hpt_pretrained`, or `filmunet_pretrained`
- `SEGMENT_LIST`: backend crop length sweep for ablation scripts
- `BATCH_SIZE`: train batch size
- `SUPERVISED_WEIGHT`: Route II-style supervised loss weight
- `BACKEND_WEIGHT`: backend loss weight
- `EXTRA_OVERRIDES`: additional Hydra overrides appended at the end of the command

`BACKEND_WEIGHT` is the preferred name in the ablation scripts. The old `PROXY_WEIGHT` spelling is still accepted as a compatibility fallback.

Route III specific:

- `LOSS_TYPES`: audio loss sweep
- `DDSP_CKPTS`: manually specify checkpoint per segment, example `DDSP_CKPTS="2=/path/a.pt 5=/path/b.pt"`
- `DDSP_PHASE`
- `CKPT_EPOCH`

Route IV specific:

- `SAMPLERS`: `coverage`, `mixed`, `realism`
- `LOSS_TYPES`: `smooth_l1`, `l1`, `mse`
- `WARMUP_ITERS`
- `PROXY_CKPT`: pass a checkpoint path as the first script argument

## Batch Size

There are two different defaults to keep in mind.

- global config default: `exp.batch_size=12`
- current Route III / IV ablation script default: `BATCH_SIZE=4`

So when you launch through [`train_route3_ablation.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_route3_ablation.sh) or [`train_route4_ablation.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_route4_ablation.sh), the effective batch size is currently `4`, not `12`.

If your local 3090 / 5090 still has a lot of free memory, the simplest way to test a larger batch is:

```bash
TRAIN_SET=maestro BATCH_SIZE=6 bash run_scripts/train_route3_ablation.sh
```

or

```bash
TRAIN_SET=maestro BATCH_SIZE=8 bash run_scripts/train_route4_ablation.sh
```

I would increase conservatively in the order `4 -> 6 -> 8`, because Route III and Route IV can have different backend-side memory spikes.

## Local BSSL / BSTL Evaluation

Default behavior:

- `train_eval.audio_metrics.enabled=false`
- `train_eval.audio_metrics.instrument_path=""`

This is intentional. Do not change the default if your main target is Kaya.

### Piano runs without BSSL

Route III:

```bash
TRAIN_SET=maestro SEGMENT_LIST="2" LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
  bash run_scripts/train_route3_ablation.sh
```

Route IV:

```bash
TRAIN_SET=maestro SEGMENT_LIST="2" SAMPLERS="mixed" LOSS_TYPES="smooth_l1" \
  bash run_scripts/train_route4_ablation.sh
```

### Guitar runs without BSSL

Route III:

```bash
TRAIN_SET=francoisleduc SEGMENT_LIST="2" LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
  bash run_scripts/train_route3_ablation.sh
```

Route IV:

```bash
TRAIN_SET=gaps SEGMENT_LIST="2" SAMPLERS="mixed" LOSS_TYPES="smooth_l1" \
  bash run_scripts/train_route4_ablation.sh
```

### Piano runs with local BSSL / BSTL evaluation

Route III:

```bash
TRAIN_SET=maestro SEGMENT_LIST="2" LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
  EXTRA_OVERRIDES="train_eval.audio_metrics.enabled=true train_eval.audio_metrics.instrument_path=/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz" \
  bash run_scripts/train_route3_ablation.sh
```

Route IV:

```bash
TRAIN_SET=maestro SEGMENT_LIST="2" SAMPLERS="mixed" LOSS_TYPES="smooth_l1" \
  EXTRA_OVERRIDES="train_eval.audio_metrics.enabled=true train_eval.audio_metrics.instrument_path=/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz" \
  bash run_scripts/train_route4_ablation.sh
```

### Guitar runs with local BSSL / BSTL evaluation

Route III:

```bash
TRAIN_SET=francoisleduc SEGMENT_LIST="2" LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
  EXTRA_OVERRIDES="train_eval.audio_metrics.enabled=true train_eval.audio_metrics.instrument_path=/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SpanishClassicalGuitar/SpanishClassicalGuitar-20190618.sfz" \
  bash run_scripts/train_route3_ablation.sh
```

Route IV:

```bash
TRAIN_SET=gaps SEGMENT_LIST="2" SAMPLERS="mixed" LOSS_TYPES="smooth_l1" \
  EXTRA_OVERRIDES="train_eval.audio_metrics.enabled=true train_eval.audio_metrics.instrument_path=/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SpanishClassicalGuitar/SpanishClassicalGuitar-20190618.sfz" \
  bash run_scripts/train_route4_ablation.sh
```

Notes:

- keep `include_train=false` unless you really want training-split audio rendering too
- `backend=auto` is fine here; `.sfz` will resolve to `sfizz`
- if local evaluation is slow, add `train_eval.audio_metrics.max_segments=1` to `EXTRA_OVERRIDES`

## Route III Wandb Charts

Route III uses differentiable audio supervision. The train-side charts mainly describe audio matching, while the eval-side charts describe velocity quality.

| Chart name | Meaning | Better | Priority |
| --- | --- | --- | --- |
| `train_total_loss` | Total optimized loss for the step average. In default Route III this is usually almost the same as `train_backend_loss`. | Smaller | High |
| `train_supervised_loss` | Raw Route II-style supervised velocity loss, only present when `loss.supervised_weight > 0`. | Smaller | High |
| `train_supervised_weighted` | Actual weighted supervised contribution added into `train_total_loss`. | Smaller | High |
| `train_backend_loss` | Main differentiable backend loss actually optimized by Route III. | Smaller | High |
| `train_backend_weighted` | Actual weighted backend contribution added into `train_total_loss`. | Smaller | High |
| `train_backend_spectral_raw` | Raw Piano-SSM spectral loss before weighting. This is the main default audio loss term. | Smaller | High |
| `train_backend_spectral_weighted` | Weighted spectral contribution after multiplying by the spectral weight. Default weight is usually `1.0`, so it often matches `spectral_raw`. | Smaller | Medium |
| `train_backend_log_rms_raw` | Raw clip-level log-RMS auxiliary loss. It helps stabilize overall loudness matching. | Smaller | Medium |
| `train_backend_log_rms_weighted` | Weighted log-RMS contribution after multiplying by `log_rms_weight`. Default weight is small, so this is usually much smaller than `log_rms_raw`. | Smaller | Medium |
| `train_backend_log_rms_log_rms_loss` | Internal log-RMS loss reported by the submodule. Semantically similar to the previous log-RMS term. | Smaller | Low |
| `train_backend_log_rms_log_rms_pred` | Average log-RMS of rendered prediction audio. | Closer to target | Low |
| `train_backend_log_rms_log_rms_target` | Average log-RMS of target audio. This is a reference trace, not an optimization score by itself. | Reference | Low |
| `train_backend_spectral_audio_rms_pred` | Average RMS of rendered prediction audio reported from the spectral-loss branch. | Closer to target | Low |
| `train_backend_spectral_audio_rms_target` | Average RMS of target audio reported from the spectral-loss branch. | Reference | Low |
| `train_backend_log_rms_audio_rms_pred` | Average RMS of rendered prediction audio reported from the log-RMS branch. | Closer to target | Low |
| `train_backend_log_rms_audio_rms_target` | Average RMS of target audio reported from the log-RMS branch. | Reference | Low |
| `velocity_roll_comparison` | Visualization of target and predicted velocity rolls. This is for visual inspection, not for scalar ranking. | Visual | High |
| `train_stat.frame_max_error` | Train-split velocity error using frame-wise max picking. | Smaller | Medium |
| `train_stat.frame_max_std` | Dispersion of `frame_max_error`. | Smaller | Medium |
| `train_stat.onset_masked_error` | Train-split velocity error measured only at onset-mask positions. Usually easier to interpret for note attacks. | Smaller | High |
| `train_stat.onset_masked_std` | Dispersion of `onset_masked_error`. | Smaller | Medium |
| `valid_<eval_set>_stat.frame_max_error` | Validation velocity error on one configured eval split using frame-wise max picking. For piano defaults this usually appears as `valid_maestro_stat.*` and `valid_smd_stat.*`. | Smaller | High |
| `valid_<eval_set>_stat.frame_max_std` | Dispersion of the eval-split frame-max error. | Smaller | Medium |
| `valid_<eval_set>_stat.onset_masked_error` | Validation velocity error on one configured eval split at onset-mask positions. | Smaller | High |
| `valid_<eval_set>_stat.onset_masked_std` | Dispersion of the eval-split onset-masked error. | Smaller | Medium |
| `*_stat.real_pred_bssl_pearson_correlation` | Optional rendered-audio Pearson metric using the BSSL representation. Appears only when `train_eval.audio_metrics.enabled=true`. | Larger | Medium |
| `*_stat.real_pred_bstl_pearson_correlation` | Optional rendered-audio Pearson metric using the BSTL representation. Appears only when `train_eval.audio_metrics.enabled=true`. | Larger | Medium |

Notes:

- `spectral_mag_*` and `spectral_logmag_*` are intentionally filtered out and should no longer appear in charts
- some backend constants are also filtered out of charts and moved to wandb summary plus local log files

## Route IV Wandb Charts

Route IV uses a frozen SFProxy model for note-wise weak supervision. The train-side charts mainly describe proxy note-feature matching, while the eval-side charts still describe velocity quality.

| Chart name | Meaning | Better | Priority |
| --- | --- | --- | --- |
| `train_total_loss` | Total optimized loss for the step average. In default Route IV this is usually almost the same as `train_backend_loss`. | Smaller | High |
| `train_supervised_loss` | Raw Route II-style supervised velocity loss, only present when `loss.supervised_weight > 0`. | Smaller | High |
| `train_supervised_weighted` | Actual weighted supervised contribution added into `train_total_loss`. | Smaller | High |
| `train_backend_loss` | Main differentiable backend loss actually optimized by Route IV. Its formula depends on `proxy.sfproxy.loss_type`. | Smaller | High |
| `train_backend_weighted` | Actual weighted backend contribution added into `train_total_loss`. | Smaller | High |
| `train_backend_note_mae` | Mean absolute error between predicted note features and target note features on valid note positions only. This is a backend note-feature MAE, not raw MIDI velocity MAE. | Smaller | High |
| `train_backend_note_count` | Average number of valid notes in the current batch crop. This is context, not a quality metric. | Context only | Low |
| `velocity_roll_comparison` | Visualization of target and predicted velocity rolls. Useful for quick sanity checking. | Visual | High |
| `train_stat.frame_max_error` | Train-split velocity error using frame-wise max picking. | Smaller | Medium |
| `train_stat.frame_max_std` | Dispersion of `frame_max_error`. | Smaller | Medium |
| `train_stat.onset_masked_error` | Train-split velocity error measured only at onset-mask positions. | Smaller | High |
| `train_stat.onset_masked_std` | Dispersion of `onset_masked_error`. | Smaller | Medium |
| `valid_<eval_set>_stat.frame_max_error` | Validation velocity error on one configured eval split using frame-wise max picking. For piano defaults this usually appears as `valid_maestro_stat.*` and `valid_smd_stat.*`. | Smaller | High |
| `valid_<eval_set>_stat.frame_max_std` | Dispersion of the eval-split frame-max error. | Smaller | Medium |
| `valid_<eval_set>_stat.onset_masked_error` | Validation velocity error on one configured eval split at onset-mask positions. | Smaller | High |
| `valid_<eval_set>_stat.onset_masked_std` | Dispersion of the eval-split onset-masked error. | Smaller | Medium |
| `*_stat.real_pred_bssl_pearson_correlation` | Optional rendered-audio Pearson metric. Appears only when `train_eval.audio_metrics.enabled=true`. | Larger | Medium |
| `*_stat.real_pred_bstl_pearson_correlation` | Optional rendered-audio Pearson metric. Appears only when `train_eval.audio_metrics.enabled=true`. | Larger | Medium |

How Route IV train metrics are computed:

- `train_backend_loss` is computed with the selected backend loss: `smooth_l1`, `l1`, or `mse`
- `train_backend_note_mae` is always mean absolute error on valid masked notes, regardless of which backend loss is used
- `train_backend_note_count` is the average valid note count in the current batch crop

## What To Focus On During Ablation

### Route III

Look at these first:

- `train_backend_loss`
- `train_backend_spectral_raw`
- `train_supervised_weighted` when you are doing mixed supervision
- `train_backend_weighted` when you are doing mixed supervision
- `valid_*_stat.onset_masked_error`
- `valid_*_stat.onset_masked_std`
- `valid_*_stat.frame_max_error`
- optional `*_pearson_correlation` if audio eval is enabled

Use these mostly as diagnostics:

- `train_backend_log_rms_raw`
- `train_backend_log_rms_weighted`
- all `*_pred` and `*_target` RMS traces

Practical reading:

- if spectral loss goes down but validation velocity error does not improve, the proxy objective may be matching audio without improving velocity semantics
- if `pred` loudness is far from `target`, the model may be learning the wrong overall energy scale

### Route IV

Look at these first:

- `train_backend_loss`
- `train_backend_note_mae`
- `train_supervised_weighted` when you are doing mixed supervision
- `train_backend_weighted` when you are doing mixed supervision
- `valid_*_stat.onset_masked_error`
- `valid_*_stat.onset_masked_std`
- `valid_*_stat.frame_max_error`

Use this mostly as context:

- `train_backend_note_count`

Practical reading:

- if `note_mae` improves but validation velocity error does not, the proxy note feature may be too weakly aligned with the downstream velocity objective
- if `note_count` changes a lot across runs, small fluctuations in train loss may partly come from crop difficulty rather than model quality

## Offline Logs

Wandb is not the only source of truth. Every training run also writes local text files inside the checkpoint folder.

- `training_stats.txt`: static run configuration summary
- `training.log`: per-eval text log with train metrics, eval metrics, and summary-only backend constants

These files live under:

```text
score_hpt/workspaces/checkpoints/<model_name>/
```

This is the easiest file to paste into Codex when you want to discuss results without relying on the wandb UI.

## Summary-Only Backend Constants

The following backend constants are intentionally removed from wandb charts. They go to wandb summary and local `training.log` instead.

- `backend_loss_sample_rate`
- `backend_loss_frame_rate`
- `backend_proxy_frames`
- `backend_proxy_polyphony`
- `backend_renderer_sample_rate`
- `backend_renderer_frame_rate`
- `backend_renderer_segment_seconds`
- `backend_synth_input_source_fps`
- `backend_renderer_hop_length`
- `backend_renderer_n_fft`
- `backend_native_segment_seconds`
