# SFProxy

This repository is now trimmed to the SoundFont proxy workflow we actually use.

It covers one project only:
- export note-conditioned teacher data from `.sf2` or `.sfz` instruments
- train a neural proxy on note-wise dynamics targets
- evaluate monotonicity and velocity-recovery behavior

## Repo layout

- `src/`: the full runtime codebase
- `configs/`: Hydra configs for export, training, and evaluation
- `tests/`: focused tests for the current proxy pipeline
- `Sfproxy_Eval.md` and `Sfproxy_Eval_zh.md`: evaluation notes

## Install

System tools:
- `fluidsynth` for `.sf2`
- `sfizz_render` for `.sfz` (optional unless you use SFZ instruments)

Python:

```bash
python -m pip install -r requirements.txt
```

## Typical workflow

Export teacher data:

```bash
python src/export_dataset_pkl.py \
  --config-name data_piano \
  dataset_size=20000 \
  start_index=0 end_index=20000
```

Train:

```bash
python src/train.py \
  --config-name train \
  dataset.train.path=/path/to/export_train_folder \
  dataset.val.path=/path/to/export_val_folder
```

Evaluate:

```bash
python src/eval.py \
  --config-name eval \
  mode=monotonic \
  ckpt_path=/path/to/checkpoint.ckpt
```

```bash
python src/eval.py \
  --config-name eval \
  mode=velocity_recovery \
  ckpt_path=/path/to/checkpoint.ckpt
```

## Notes

- The default Hydra paths still point at the shared workspace used in your current setup.
- Override `paths.workspace_dir`, dataset paths, or instrument paths from the CLI if you want to relocate outputs.
- `configs/data_piano.yaml` and `configs/data_guitar.yaml` hold the current sampler presets.
