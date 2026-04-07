# Route I direct inversion + shared evaluation

This folder contains the Route I baseline and the shared evaluation utilities used by Route I / II / III / IV.

## Files

- `note_loudness.py`
  - Extracts the two note-wise loudness parameters used in the paper draft:
    - harmonic-energy-after-onset
    - onset spectral flux
- `midi_velocity_mapping.py`
  - Dataset-statistics MIDI velocity mapping
  - Converts note-wise loudness features into percentile-like scores and then into MIDI velocities
- `midi_tools.py`
  - MIDI note loading, velocity replacement, flat-velocity baseline, note-wise velocity MAE alignment
- `route1.py`
  - Route I batch runner
  - Creates both direct-inversion and flat-velocity predicted MIDIs plus manifests
- `eval_framework.py`
  - Shared rendering + evaluation backend
  - Computes:
    - velocity MAE
    - BSSL Pearson / MAE / cosine / Spearman
    - BSTL Pearson / MAE / cosine / Spearman
  - Supports both:
    - `SoundFont(gt note, pred velocity)` vs real audio
    - `SoundFont(gt note, pred velocity)` vs `SoundFont(gt note, gt velocity)`

## CLI examples

### Route I dataset inference

```bash
cd score_hpt
python pytorch/direct_invension/route1.py dataset \
  --dataset_type smd \
  --dataset_dir /path/to/SMD \
  --stats_json ../data_analysis/stats/midi_sampler/SMD_sampler.json \
  --out_dir ./outputs/route1_smd
```

### Evaluate an existing manifest

```bash
cd score_hpt
python pytorch/direct_invension/eval_framework.py manifest \
  --manifest ./outputs/route1_smd/route1_direct_manifest.json \
  --instrument /path/to/soundfont.sf2 \
  --out_dir ./outputs/route1_smd/eval
```

### Build a manifest from a predicted MIDI folder and evaluate it

```bash
cd score_hpt
python pytorch/direct_invension/eval_framework.py dataset \
  --dataset_type smd \
  --dataset_dir /path/to/SMD \
  --pred_midi_dir /path/to/pred_midis \
  --label route2_veloest \
  --instrument /path/to/soundfont.sf2 \
  --out_dir ./outputs/route2_eval
```
