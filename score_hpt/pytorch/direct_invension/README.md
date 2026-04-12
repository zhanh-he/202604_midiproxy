# Route I direct inversion

This folder keeps the Route I / II implementations, the flat-velocity baseline, and the shared evaluation backend.

## Files

- `route1_infer.py`
  - Route I inference implementation
  - note-wise loudness extraction
  - dataset-statistics velocity mapping
  - same-structure MIDI export
- `flat_infer.py`
  - fixed-velocity MIDI export
  - dataset scan matching Route I CLI behavior
- `route2_infer.py`
  - Route II model inference
  - aligned score-note velocity backfill to prediction MIDI
- `route1_evaluate.py`
  - Route I evaluation entrypoint
- `route2_evaluate.py`
  - Route II evaluation entrypoint
- `flat_evaluate.py`
  - flat-velocity evaluation entrypoint
- `eval_runner.py`
  - shared evaluation runner for Route I / flat baselines
- `common.py`
  - shared config/path/JSON helpers
  - shared PrettyMIDI note sorting and velocity rewriting
- `eval_framework.py`
  - shared rendering and evaluation backend
  - reusable by Route II / III / IV

## Public CLI entrypoints

Run these from `score_hpt/`:

```bash
python pytorch/direct_invension/route1_infer.py dataset.test_set=smd
python pytorch/direct_invension/route1_evaluate.py dataset.test_set=smd route1.eval.instrument_path=/path/to/soundfont.sf2
python pytorch/direct_invension/route2_infer.py dataset.test_set=smd model.frontend_pretrained=/path/to/ckpt.pth
python pytorch/direct_invension/route2_evaluate.py dataset.test_set=smd route2.eval.instrument_path=/path/to/soundfont.sf2
python pytorch/direct_invension/flat_infer.py dataset.test_set=smd flat.infer.flat_velocity=64
python pytorch/direct_invension/flat_evaluate.py dataset.test_set=smd flat.eval.instrument_path=/path/to/soundfont.sf2
```

## Config source

Defaults come from:

```text
pytorch/config/config.yaml
```

Important Route I defaults:

- `feature.frames_per_second=100`
- `backend.supervision.hop_size=221`
- `route1.infer.*`
- `route1.eval.*`

Important flat defaults:

- `flat.infer.*`
- `flat.eval.*`

Important Route II defaults:

- `route2.infer.*`
- `route2.eval.*`

## Folder-mode evaluation

Use folder mode when GT MIDI and/or reference audio do not follow the built-in dataset layout.

```bash
python pytorch/direct_invension/route1_evaluate.py \
  dataset.test_set=maestro \
  route1.eval.instrument_path=/path/to/soundfont.sf2 \
  route1.eval.manifest_mode=folder \
  route1.eval.audio_reference_mode=folder \
  route1.eval.gt_midi_dir=/path/to/gt_midis \
  route1.eval.reference_audio_dir=/path/to/reference_audio
```

## Audio reference modes

- `route1.eval.audio_reference_mode=dataset`
  - compare against dataset real audio
- `route1.eval.audio_reference_mode=folder`
  - compare against a matched external audio folder
- `route1.eval.audio_reference_mode=none`
  - skip real-audio reference and keep only synth-reference metrics
