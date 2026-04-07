"""SoundFont neural proxy pipeline.

This package adds an alternative pipeline to train differentiable neural proxies for
SoundFont-based instruments. The baseline repository trains preset->audio-embedding proxies
for VST synthesizers. Here we focus on note-event conditioning and note-wise dynamics targets.

Phase-1 scope:
- SF2 rendering via FluidSynth (Python binding if available, otherwise CLI fallback)
- SFZ rendering via sfizz_render CLI
- Offline dataset export to pickled torch tensors
- Transformer-based proxy model

"""

from __future__ import annotations
