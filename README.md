# 202604_midiproxy

This repository studies **MIDI velocity estimation from audio + MIDI score**.
The main estimator is `score_hpt/`.
Before running the route experiments, the differentiable backends are now aligned to one repo-level contract.

## Unified repo default contract

Use these defaults unless you deliberately run an ablation:

- sample rate = `22050`
- segment length = `10 s`
- target frame rate = `100 fps`
- `n_fft = 2048`
- DiffProxy hop size = `round(sample_rate / frame_rate)`

That means:

- `DDSP-Piano` default = `22050 / 10 s / 100 fps`
- `DDSP-Guitar-Synth` default = `22050 / 10 s / 100 fps target`
- `DiffProxy` default export = `22050 / 10 s / 100 fps / n_fft=2048 / hop≈221`

`score_hpt` itself stays at `22050 / 10 s / 100 fps`.
So new backend checkpoints trained with the repo defaults line up much better with later route experiments.

## Native defaults from the original projects

These are reference values only.
They are **not** the new repo defaults.

- upstream `DDSP-Piano` style default: `16000 / 3 s / 250 fps`
- upstream `DDSP-Guitar-Synth` style default:
  - sample rate = `16000`
  - hop length = `64`
  - effective frame rate = `250 fps`
  - `n_fft = 2048`
  - segment length = `3 s`
  - loss FFT sizes = `[64, 128, 256, 512, 1024, 2048]`
- earlier `DiffProxy` exports in this repo often used `2 s` note-audio segments
  - piano example: `22050 / 2 s / n_fft=2048 / hop=256`
  - guitar example: `32000 / 2 s / n_fft=2048 / hop=256`

## Repository layout

- `score_hpt/`
  - main velocity estimator
  - route mapping:
    - Route I: Note-wise Parameter Extraction
    - Route II: Score-Inf Velocity Estimation
    - Route III: Score-Inf VeloEst + DiffSynth
    - Route IV: Score-Inf VeloEst + DiffProxy
  - training entry points:
    - `pytorch/train.py`: Route II
    - `pytorch/train_ddsp.py`: Route III
    - `pytorch/train_proxy.py`: Route IV
    - `pytorch/train_backend.py`: shared implementation for Route III and Route IV
    - `pytorch/velo_model/`: base velocity-estimation models

- `synthesizer/ddsp-piano-pytorch/`
  - `DDSP-Piano` training code

- `synthesizer/ddsp-guitar-synth/`
  - **current default guitar DiffSynth backend**
  - unified CLI dataset preparation and training scripts

- `synthesizer/ddsp-guitar/`
  - older guitar DiffSynth backend
  - kept for ablation only

- `synth-proxy/`
  - `DiffProxy` export and training code

## DDSP-Piano training

Preferred repo entry point:

- `run_scripts/train_ddsp_piano.sh`

That script is now the only maintained repo-level launcher for DDSP-Piano.
Edit the manual config block at the top of the script to choose:

- machine: `3090` or `5090`
- segment length: `2`, `5`, or `10`
- total epochs

Behavior:

- if the cache directory is empty, it runs preprocessing first
- if cached data already exists, it skips preprocessing and starts training directly

### Dataset arrangement

This code uses the **official MAESTRO split from `maestro-v3.0.0.csv`**.
The preprocessing loop iterates over every row whose `split` equals `train`, `validation`, or `test`.
So yes, it uses the full MAESTRO dataset under the official split definitions.

### 10 s

```bash
cd synthesizer/ddsp-piano-pytorch
python preprocess.py \
  --splits train validation test \
  --segment_duration 10 \
  --sample_rate 22050 \
  --frame_rate 100 \
  /path/to/maestro-v3.0.0 \
  /path/to/workspaces_unified_10s/data_cache

python train.py \
  --phase 1 \
  --batch_size 6 \
  --epochs 7 \
  --lr 1e-3 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration 10 \
  /path/to/workspaces_unified_10s/data_cache \
  /path/to/workspaces_unified_10s/models
```

### 5 s

```bash
cd synthesizer/ddsp-piano-pytorch
python preprocess.py \
  --splits train validation test \
  --segment_duration 5 \
  --sample_rate 22050 \
  --frame_rate 100 \
  /path/to/maestro-v3.0.0 \
  /path/to/workspaces_unified_5s/data_cache

python train.py \
  --phase 1 \
  --batch_size 6 \
  --epochs 7 \
  --lr 1e-3 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration 5 \
  /path/to/workspaces_unified_5s/data_cache \
  /path/to/workspaces_unified_5s/models
```

### 2 s

```bash
cd synthesizer/ddsp-piano-pytorch
python preprocess.py \
  --splits train validation test \
  --segment_duration 2 \
  --sample_rate 22050 \
  --frame_rate 100 \
  /path/to/maestro-v3.0.0 \
  /path/to/workspaces_unified_2s/data_cache

python train.py \
  --phase 1 \
  --batch_size 6 \
  --epochs 7 \
  --lr 1e-3 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration 2 \
  /path/to/workspaces_unified_2s/data_cache \
  /path/to/workspaces_unified_2s/models
```

## DDSP-Guitar-Synth training

This backend is now the default guitar DiffSynth choice in the repo.
The repo-level launcher is now:

- `run_scripts/train_ddsp_guitar_synth.sh`

Edit the manual config block at the top of that script to choose:

- machine: `3090` or `5090`
- segment length: `2`, `5`, or `10`
- total epochs

Behavior:

- if prepared `.npz` datasets are missing, it runs data preparation first
- if prepared datasets already exist, it skips preparation and starts training directly
- training logs to `wandb`

### What changed

New default contract:

- sample rate = `22050`
- target frame rate = `100 fps`
- derived hop length = `221`
- effective frame rate = `22050 / 221 = 99.7738 fps`
- `n_fft = 2048`
- default segment length = `10 s`
- default loss FFT sizes = `[128, 256, 512, 1024, 2048]`

The original project was GuitarSet-based through `mirdata`.
The new helper scripts keep that arrangement.
Default split:

- train = all players except `00`
- validation = player `00`

### 10 s

```bash
cd synthesizer/ddsp-guitar-synth
python prepare_guitarset_unified.py \
  --guitarset_path /path/to/GuitarSet \
  --output_dir /path/to/ddsp_guitar_synth_10s/data \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 10

python train_midi_synth_unified.py \
  --train_dataset_path /path/to/ddsp_guitar_synth_10s/data/train_gset_midi_10s.npz \
  --val_dataset_path /path/to/ddsp_guitar_synth_10s/data/val_gset_midi_10s.npz \
  --output_dir /path/to/ddsp_guitar_synth_10s/output \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 10 \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
```

### 5 s

```bash
cd synthesizer/ddsp-guitar-synth
python prepare_guitarset_unified.py \
  --guitarset_path /path/to/GuitarSet \
  --output_dir /path/to/ddsp_guitar_synth_5s/data \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 5

python train_midi_synth_unified.py \
  --train_dataset_path /path/to/ddsp_guitar_synth_5s/data/train_gset_midi_5s.npz \
  --val_dataset_path /path/to/ddsp_guitar_synth_5s/data/val_gset_midi_5s.npz \
  --output_dir /path/to/ddsp_guitar_synth_5s/output \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 5 \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
```

### 2 s

```bash
cd synthesizer/ddsp-guitar-synth
python prepare_guitarset_unified.py \
  --guitarset_path /path/to/GuitarSet \
  --output_dir /path/to/ddsp_guitar_synth_2s/data \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 2

python train_midi_synth_unified.py \
  --train_dataset_path /path/to/ddsp_guitar_synth_2s/data/train_gset_midi_2s.npz \
  --val_dataset_path /path/to/ddsp_guitar_synth_2s/data/val_gset_midi_2s.npz \
  --output_dir /path/to/ddsp_guitar_synth_2s/output \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 2 \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
```

Project-local helper scripts are also available if you want to run preparation and training separately:

```bash
bash synthesizer/ddsp-guitar-synth/run_prepare_unified.sh 10 /path/to/GuitarSet /path/to/ddsp_guitar_synth_10s/data
bash synthesizer/ddsp-guitar-synth/run_training_unified.sh 10 /path/to/ddsp_guitar_synth_10s/data /path/to/ddsp_guitar_synth_10s/output
```

## Switch back to legacy `ddsp-guitar`

For ablation, keep `proxy.type=diffsynth_guitar` and switch only the implementation:

```yaml
proxy:
  type: diffsynth_guitar
  checkpoint: /path/to/legacy_ddsp_guitar.ckpt
  ddsp:
    project_root: /abs/path/to/202604_midiproxy/synthesizer/ddsp-guitar
  ddsp_guitar:
    implementation: ddsp_guitar
```

New default:

```yaml
proxy:
  type: diffsynth_guitar
  checkpoint: /path/to/ddsp_guitar_synth/latest_model_checkpoint.pt
  ddsp:
    project_root: /abs/path/to/202604_midiproxy/synthesizer/ddsp-guitar-synth
  ddsp_guitar:
    implementation: ddsp_guitar_synth
```

That keeps the ablation minimal.
Only the guitar DiffSynth implementation changes.
