# Score-HPT + Backend Supervision

This note is the current training reference for `score_hpt/`.
It keeps only the paths and options that still match the codebase.

## What this project trains

Inside [`pytorch/`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch):

- `train.py`
  - Route II
  - supervised velocity estimation
- `train_ddsp.py`
  - Route III
  - velocity estimation with DiffSynth backend loss
- `train_proxy.py`
  - Route IV
  - velocity estimation with frozen SFProxy backend loss
- `train_backend.py`
  - shared Route III / IV implementation

## Model combinations

The project is intentionally narrowed to three main combinations:

- `filmunet-direct`
- `hpt-direct`
- `hpt-note_editor`

Operational rules:

- `filmunet` is always run with `score_informed.method=direct`
- `hpt` may use `direct` or `note_editor`
- `model.frontend_pretrained=""` means scratch training
- `model.frontend_pretrained=/path/to/ckpt` means load and continue

FiLMUNet notes:

- FiLMUNet is fixed to frame conditioning in this repo
- no extra conditioning variant is part of the current main path

## Current losses

### Route II

Available supervised velocity losses:

- `velocity_bce`
- `velocity_mse`
- `kim_bce_l1`

Recommended default:

- `kim_bce_l1`

### Route III

Current Route III backend losses:

- `piano_ssm_combined`
- `piano_ssm_combined_rm`
- `piano_ssm_spectral`
- `piano_ssm_spectral_plus_log_rms`
- `piano_ssm_spectral_plus_diffsynth_loudness`
- `diffsynth_piano_loudness`

Recommended default:

- `piano_ssm_spectral_plus_log_rms`

Useful comparison set:

- `piano_ssm_spectral`
- `piano_ssm_spectral_plus_log_rms`
- `piano_ssm_spectral_plus_diffsynth_loudness`
- `piano_ssm_combined_rm`

### Route IV

Current Route IV backend losses:

- `smooth_l1`
- `l1`
- `mse`

Recommended default:

- `smooth_l1`

Recommended order for ablation:

1. `smooth_l1`
2. `l1`
3. `mse`

### Prior loss

The optional auxiliary regularizer is:

- `velocity_prior_loss`

Use it as a small regularizer only.
It is not the main supervision path.

## Suggested ablation order

To keep experiments interpretable:

1. fix the model combination
2. scan Route III backend losses
3. scan Route IV backend losses
4. only then compare model combinations or supervised/backend mixing

This keeps one moving part at a time.

## Wandb logging

### Run names

Route III / IV runs now encode the main experiment identity directly in `wandb.name`.

Typical forms are:

- `route3-maestro-2s-hpt-note_editor-...`
- `route3-maestro-2s-hpt-direct-...`
- `route3-maestro-2s-filmunet-direct-...`
- `route4-maestro-5s-hpt-note_editor-...`

The backend segment length is part of the run name.

### Main charts

Train-side common metrics:

- `train_total_loss`
- `train_supervised_loss`
- `train_backend_loss`
- `train_supervised_weighted`
- `train_backend_weighted`
- `train_prior_loss`

Validation-side common metrics:

- `train_stat.frame_max_error`
- `train_stat.frame_max_std`
- `train_stat.onset_masked_error`
- `train_stat.onset_masked_std`
- `valid_maestro_stat.*`
- `valid_smd_stat.*`

Image logging:

- `velocity_roll_comparison`

### Route III extra backend charts

When Route III uses the default `piano_ssm_spectral_plus_log_rms`, train-side charts also include:

- `train_backend_spectral_raw`
- `train_backend_spectral_weighted`
- `train_backend_log_rms_raw`
- `train_backend_log_rms_weighted`
- `train_backend_log_rms_log_rms_loss`
- `train_backend_log_rms_log_rms_pred`
- `train_backend_log_rms_log_rms_target`
- `train_backend_*audio_rms_*`

### Route IV extra backend charts

Route IV commonly adds:

- `train_backend_note_mae`
- `train_backend_note_count`

### Summary-only constants

These no longer go into charts.
They are kept in wandb summary and local logs instead:

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

## Local log files

Each Route III / IV checkpoint directory writes:

- `training_stats.txt`
- `training.log`

`training.log` is the most useful file for later discussion because it gives text snapshots of:

- train aggregates
- eval aggregates
- summary constants

## BSSL / BSTL evaluation

Rendered-audio Pearson evaluation is available but stays off by default.

Control it with:

- `train_eval.audio_metrics.enabled`
- `train_eval.audio_metrics.instrument_path`

Default behavior:

- disabled
- no instrument path

That is intentional, so training stays safe on Kaya and on machines without local soundfonts.

When you explicitly enable it on a local machine with `sfz` or `sf2` support, wandb will additionally log:

- `real_pred_bssl_pearson_correlation`
- `real_pred_bstl_pearson_correlation`
