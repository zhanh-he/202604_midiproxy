# 202604_midiproxy

This repository studies MIDI velocity estimation from audio plus symbolic score.

The repo currently has one main research path:

- `score_hpt/`
  - Route I: note-wise parameter extraction
  - Route II: supervised velocity estimation
  - Route III: velocity estimation with DiffSynth backend supervision
  - Route IV: velocity estimation with DiffProxy backend supervision

Supporting components:

- `synth-proxy/`
  - SoundFont teacher-data export
  - SFProxy training
- `synthesizer/ddsp-piano-pytorch/`
  - DDSP piano backend training
- `synthesizer/ddsp-guitar-synth/`
  - DDSP guitar backend training
- `run_scripts/`
  - local training and ablation helpers
- `kaya_scripts/`
  - Kaya cluster launchers

## Current repo defaults

Unless you are deliberately running an ablation, the main repo contract is:

- sample rate: `22050`
- front-end segment length: `10 s`
- frame rate: `100 fps`
- `n_fft=2048`

Route III and Route IV now keep the Score-HPT front-end fixed at `10 s`.
Backend crop length is ablated separately through `backend.backend_segment_seconds`.

## Main docs

Use these files as the current source of truth:

- [`score_hpt/README_scoreinf_proxy.md`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/README_scoreinf_proxy.md)
  - model combinations
  - losses
  - wandb metrics
  - route-level training notes
- [`run_scripts/train_route.md`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_route.md)
  - local Route III and Route IV commands
  - BSSL / BSTL local evaluation
  - ablation examples
- [`kaya_scripts/README.md`](/media/mengh/SharedData/zhanh/202604_midiproxy/kaya_scripts/README.md)
  - Kaya data layout
  - Kaya submission commands
  - Kaya sweep contents

## Main training entry points

Inside [`score_hpt/pytorch/`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch):

- `train.py`
  - Route II
- `train_ddsp.py`
  - Route III
- `train_proxy.py`
  - Route IV
- `train_backend.py`
  - shared Route III / IV implementation

## Current model combinations

The project is intentionally narrowed to three main training combinations:

- `filmunet-direct`
- `hpt-direct`
- `hpt-note_editor`

Defaults:

- `MODEL_TYPE=hpt`
- `SCORE_METHOD=note_editor`
- `FRONTEND_PRETRAINED=""`

If you provide `FRONTEND_PRETRAINED=/path/to/ckpt`, training will load it before continuing.
If you leave it empty, training is from scratch.

## Notes

- local and Kaya scripts no longer own batch size; Hydra config does
- local and Kaya scripts no longer use separate pretrained wrapper launchers
- BSSL / BSTL evaluation is off by default and should be enabled only when you explicitly want local rendered-audio evaluation
