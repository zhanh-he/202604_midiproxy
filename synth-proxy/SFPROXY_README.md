# SFProxy (SoundFont Neural Proxy)

This repository originally trains preset-to-audio proxies for VST synthesizers.

This patch adds a note-conditioned SFProxy pipeline under `src/sfproxy/`.
It supports both `.sf2` and `.sfz` instrument files.

## What is added

- `src/sfproxy/`:
  - `export_dataset_pkl.py`: offline teacher dataset export (note sampler + instrument render + dynamics target extraction)
  - `train.py`: Lightning training script for the note proxy
  - `eval.py`: unified evaluation entrypoint
  - `eval_monotonic.py`: monotonicity sweep tool
  - `eval_velocity_recovery.py`: gradient-based velocity recovery evaluation
  - `renderers/fluidsynth_sf2.py`: SF2 renderer (pyfluidsynth binding or fluidsynth CLI fallback)
  - `renderers/sfizz_sfz.py`: SFZ renderer (`sfizz_render` CLI)
  - `features/dynamics.py`: note-wise targets (harmonic energy + onset flux)
  - `data/`: samplers, datasets

- `configs/sfproxy/`:
  - `train.yaml`: training config
  - `data.yaml`: shared export config base
  - `data_piano.yaml`: piano export + sampler config
  - `data_guitar.yaml`: guitar export + sampler config
  - `eval.yaml`: shared eval config for monotonic and velocity recovery

## Quick start

### 0) Requirements

- Install FluidSynth
  - Linux: `sudo apt install fluidsynth`
  - macOS: `brew install fluidsynth`

- Python deps (at least): `hydra-core`, `lightning`, `torch`, `mido`, `pyfluidsynth`.

### 1) Prepare instrument files

Place your `.sf2` or `.sfz` files, for example:

- `${PROJECT_ROOT}/data/soundfonts/piano.sf2`
- `${PROJECT_ROOT}/data/soundfonts/guitar.sf2`

### 2) Export teacher datasets

Train set (piano example):

```bash
python src/sfproxy/export_dataset_pkl.py \
  --config-name data_piano \
  dataset_size=20000 \
  start_index=0 end_index=20000 \
  batch_size=4 num_workers=0
```

Sampler ablation for the same instrument preset:

```bash
python src/sfproxy/export_dataset_pkl.py \
  --config-name data_piano \
  sampler_preset=realism \
  sampler_velo_present=true \
  dataset_size=20000
```

Val set (use different seed_offset):

```bash
python src/sfproxy/export_dataset_pkl.py \
  --config-name data_piano \
  dataset_size=2000 \
  seed_offset=1 \
  start_index=0 end_index=2000
```

Each run creates a timestamped folder under:

- `${paths.workspace_dir}/${paths.project_name}/teacher_data/<instrument>/export_YYYY-mm-dd_HH-MM-SS/`

That folder contains `configs.pkl`, `inputs_*.pkl`, `targets_*.pkl`.

Path defaults are configured directly in:

- `configs/sfproxy/data.yaml`
- `configs/sfproxy/train.yaml`

The instrument-specific sampling setup lives in:

- `configs/sfproxy/data_piano.yaml`
- `configs/sfproxy/data_guitar.yaml`

Each export preset now exposes multiple sampler options inside the same
instrument config. By default `sampler_preset=mixed_v2`; switch to
`sampler_preset=realism_v2` to run ablations without changing files. Use
`sampler_velo_present=true` only when your realism stats JSON includes a
reliable `velocities_01` distribution; otherwise realism falls back to the
hand-crafted velocity sampler.

### 2.4) Sampler presets and chord-velocity modes

The export presets keep old and new Route IV teacher-data assumptions side by
side so sampler ablations do not require code edits.

- `coverage_shared_legacy`: the old baseline. It covers pitch / duration / IOI
  broadly, but every note inside one chord onset shares the same velocity.
- `coverage_v2` and `realism_v2`: updated note-wise samplers. Chord velocities
  can be `shared`, `independent`, `correlated`, or `mixed`. In the default
  `mixed` mode, shared / independent / correlated chord draws are mixed with
  configurable probabilities.
- `mixed_v2`: the default curriculum mixture. It combines a boundary-focused
  low-polyphony sampler, a broader coverage sampler, a realism sampler driven by
  corpus histograms, and a stress sampler with denser overlap.

SoundFont-aware boundaries are optional. When `velocity_boundary_path` points to
a JSON produced by `discover_velocity_boundaries.py`, the sampler biases more
velocity draws around the discovered response-change locations of that
SoundFont. If no boundary JSON is provided, sampling falls back to the fixed
legacy boundaries at 0.33 and 0.66.

By default this project writes processed teacher data and trained proxy runs under:

- `/media/mengh/SharedData/zhanh/202601_midisemi_data/synth-proxy_v1/`

To change the storage root on another machine, override:

```bash
paths.workspace_dir=/your/data/root
```

### 2.5) Build realism sampler stats

The realism sampler reads compact MIDI statistics JSON files generated from a
real dataset. Use the sibling `data_analysis` project:

```bash
cd /media/mengh/SharedData/zhanh/202601_midisemi/data_analysis
python dataset_midi_stats.py \
  --dataset MAESTRO_v3.0.0 \
  --root /path/to/maestro \
  --json-out-dir stats/midi_sampler \
  --output-dir figures/midi_sampler_stats \
  --no-show
```

For guitar, run the same script on GuitarSet and keep the dataset name as
`GuitarSet` so the default config path resolves to `stats/midi_sampler/GuitarSet_sampler.json`.

### 3) Train the proxy

```bash
python src/sfproxy/train.py \
  --config-name train \
  dataset.train.path=/path/to/export_train_folder \
  dataset.val.path=/path/to/export_val_folder
```

### 4) Monotonic sweep (sanity check)

```bash
python src/sfproxy/eval.py \
  --config-name eval \
  mode=monotonic \
  ckpt_path=/path/to/checkpoint.ckpt
```

### 5) Velocity recovery test (in-domain + stress)

This test evaluates whether the proxy provides *useful gradients* to recover
per-note velocities from target features.

It generates synthetic segments with ground-truth velocities, renders audio
with the instrument file, extracts target dynamics features, and then optimizes
velocities through the proxy to match the targets.

```bash
python src/sfproxy/eval.py \
  --config-name eval \
  mode=velocity_recovery \
  ckpt_path=/path/to/checkpoint.ckpt \
  device=cuda \
  velocity_recovery.indomain.num_segments=50 \
  velocity_recovery.stress.num_segments=50
```

The script writes a JSON report and a checkpoint-named PNG summary to the Hydra output directory:

- `velocity_recovery_results.json`
- `<checkpoint_stem>_velocity_recovery_summary.png`

Velocity error metrics in that JSON are reported in MIDI velocity units
`[0,127]` and also include normalized `[0,1]` variants plus an `init`
baseline for comparison.

You can visualize a report with:

```bash
python src/sfproxy/eval_velocity_recovery.py \
  --plot /path/to/velocity_recovery_results.json
```

## Notes

- SF2 rendering uses:
  - Python binding `fluidsynth` (pyfluidsynth), if available
  - otherwise `fluidsynth` CLI fallback

- Phase-1 features are simple and stable:
  - log harmonic energy (sum of first harmonics)
  - log onset flux (spectral flux around onset)

You can extend `features/dynamics.py` to add richer note targets later.
