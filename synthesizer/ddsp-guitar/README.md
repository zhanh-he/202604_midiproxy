# Legacy DDSP-Guitar

This folder is kept only for ablation.
The repo default guitar DiffSynth backend is now `synthesizer/ddsp-guitar-synth/`.


This directory trains `DDSP-Guitar` backends for later use inside `score_hpt`.

## Current repo default contract

The repo now defaults to:

- sample rate = `22050`
- segment length = `10 s`
- frame rate = `100 fps`
- spectral loss FFT sizes = `[128, 256, 512, 1024, 2048]`

Historical upstream-style defaults were closer to `48000 / 8 s / 128 fps`.
Those are now reference values only.

## Training stages

There are two training stages in this folder:

1. `train_synthesis.py`
   - trains the synthesis model
2. `train_control.py`
   - trains the control model
   - requires a synthesis checkpoint

## New CLI controls

Both scripts now support:

- `--segment_seconds 10 | 5 | 2`
- `--sample_rate 22050`
- `--frame_rate 100`
- `--train_prepared_data_path ...`
- `--val_prepared_data_path ...`
- `--gpu 0`
- `--wandb_project ...`
- `--wandb_run_name ...`

`train_control.py` also supports:

- `--synthesis_model_checkpoint /path/to/synthesis.ckpt`

## Start commands

### Synthesis model

#### 10 s

```bash
cd synthesizer/ddsp-guitar
python train_synthesis.py \
  --segment_seconds 10 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --train_prepared_data_path /path/to/fl_train.pt \
  --val_prepared_data_path /path/to/fl_val.pt \
  --gpu 0
```

#### 5 s

```bash
cd synthesizer/ddsp-guitar
python train_synthesis.py \
  --segment_seconds 5 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --train_prepared_data_path /path/to/fl_train.pt \
  --val_prepared_data_path /path/to/fl_val.pt \
  --gpu 0
```

#### 2 s

```bash
cd synthesizer/ddsp-guitar
python train_synthesis.py \
  --segment_seconds 2 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --train_prepared_data_path /path/to/fl_train.pt \
  --val_prepared_data_path /path/to/fl_val.pt \
  --gpu 0
```

### Control model

#### 10 s

```bash
cd synthesizer/ddsp-guitar
python train_control.py \
  --segment_seconds 10 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --synthesis_model_checkpoint /path/to/synthesis.ckpt \
  --train_prepared_data_path /path/to/fl_train.pt \
  --val_prepared_data_path /path/to/fl_val.pt \
  --gpu 0
```

#### 5 s

```bash
cd synthesizer/ddsp-guitar
python train_control.py \
  --segment_seconds 5 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --synthesis_model_checkpoint /path/to/synthesis.ckpt \
  --train_prepared_data_path /path/to/fl_train.pt \
  --val_prepared_data_path /path/to/fl_val.pt \
  --gpu 0
```

#### 2 s

```bash
cd synthesizer/ddsp-guitar
python train_control.py \
  --segment_seconds 2 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --synthesis_model_checkpoint /path/to/synthesis.ckpt \
  --train_prepared_data_path /path/to/fl_train.pt \
  --val_prepared_data_path /path/to/fl_val.pt \
  --gpu 0
```

## Helper scripts

- `run_synthesis_unified.sh`
- `run_control_unified.sh`

## Prepared dataset requirement

The training code still expects prepared `.pt` files in the internal serialized format used by `data.GuitarSetDataset(..., prepared_path=...)`.

Because your actual `FrancoisLeducGuitarDataset` implementation is local to your machine, this repo now includes an explicit adapter plan:

- `CODEX_FL_GUITAR_PREP_PLAN.txt`

That file tells Codex exactly what fields and shapes must be produced so the new CLI training scripts can run on your FL dataset with minimal local glue code.
