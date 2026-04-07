# `score_hpt` Routes

This note is only about the code under `score_hpt/pytorch/`.

Current route naming:

- Route I
  - Note-wise Parameter Extraction
  - separate stage
  - not handled by the training entrypoints in this folder

- Route II
  - Score-Inf Velocity Estimation

- Route III
  - Score-Inf VeloEst + DiffSynth

- Route IV
  - Score-Inf VeloEst + DiffProxy

## Scripts

- `pytorch/velo_model/`
  - base velocity-estimation models
  - renamed from the older base-model package name

- `pytorch/train.py`
  - Route II
  - supervised training only
  - no differentiable backend

- `pytorch/train_backend.py`
  - shared implementation for backend-based training
  - used by Route III and Route IV
  - usually you do not call this file directly

- `pytorch/train_ddsp.py`
  - Route III
  - backend training with `proxy.enabled=true`
  - default backend is `proxy.type=diffsynth_piano`
  - can also be overridden to `proxy.type=diffsynth_guitar`

- `pytorch/train_proxy.py`
  - Route IV
  - backend training with `proxy.enabled=true`
  - default backend is `proxy.type=diffproxy`

## Route Summary

- Route II: waveform + score -> `Score-HPT` -> velocity supervision
- Route III: waveform + score -> `Score-HPT` -> frozen `DiffSynth` -> audio-based backend loss
- Route IV: waveform + score -> `Score-HPT` -> frozen `DiffProxy` -> feature-based backend loss

In code, Route III and Route IV both go through:

```text
total_loss = supervised_loss + backend_loss + prior_loss
```

The internal tensor name is still `proxy_loss` for compatibility, but conceptually it is backend loss.

## Quick Start

Run from:

```bash
cd score_hpt
```

Route II:

```bash
python pytorch/train.py \
  model.type=hpt \
  score_informed.method=direct
```

Route III, piano DiffSynth:

```bash
python pytorch/train_ddsp.py \
  proxy.checkpoint=/path/to/ddsp_piano.ckpt
```

Route III, guitar DiffSynth:

```bash
python pytorch/train_ddsp.py \
  proxy.type=diffsynth_guitar \
  proxy.checkpoint=/path/to/ddsp_guitar.ckpt
```

Route IV, DiffProxy:

```bash
python pytorch/train_proxy.py \
  proxy.checkpoint=/path/to/diffproxy.ckpt \
  proxy.sfproxy.instrument_name=salamander_piano
```

## Names You Will See

- `DiffSynth`
  - DDSP-based differentiable synthesizer backend

- `DiffProxy`
  - differentiable proxy backend
  - in this repo this means the SFProxy family

- `synthesizer-input tensors`
  - tensors passed from `score_hpt` into the backend
  - preferred guitar keys:
    - `synth_midi_pitch`
    - `synth_string_index`
    - `synth_midi_onsets`
    - `synth_midi_activity`
  - legacy `proxy_*` aliases are still accepted

## Practical Notes

- `proxy.backend_segment_seconds=0.0`
  - use the backend native/default segment length

- `proxy.type` preferred values:
  - `diffsynth_piano`
  - `diffsynth_guitar`
  - `diffproxy`

- legacy aliases such as `ddsp_piano`, `ddsp_guitar`, `sfproxy` still work, but new commands should use the preferred names above

- current default guitar DiffSynth backend lives in:
  - `/media/mengh/SharedData/zhanh/202604_midiproxy/synthesizer/ddsp-guitar-synth`

- legacy guitar ablation backend lives in:
  - `/media/mengh/SharedData/zhanh/202604_midiproxy/synthesizer/ddsp-guitar`
