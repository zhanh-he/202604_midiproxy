from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from sfproxy.renderers.base import NoteEvent


@dataclass
class SFZConfig:
    sfz_path: str


class SFZRendererStub:
    """Stub for SFZ rendering.

    Phase-1 does not implement SFZ rendering.
    Suggested Phase-2 approach:
    - Use DawDreamer with sfizz VST3
    - Or bind libsfizz
    """

    def __init__(self, cfg: SFZConfig):
        self.cfg = cfg

    def render_segment(self, notes: List[NoteEvent], sr: int, seg_len_s: float) -> np.ndarray:
        raise NotImplementedError("SFZ rendering is not implemented in Phase-1")
