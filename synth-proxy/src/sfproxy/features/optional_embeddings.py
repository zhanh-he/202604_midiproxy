from __future__ import annotations

"""Optional segment-level embedding features.

Phase-1 keeps this module optional and disabled by default.
Future work can reuse the baseline repository audio models (src/models/audio) to
extract segment embeddings.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EmbeddingFeatureConfig:
    use_embedding: bool = False
    model_name: str = "mn04"


class SegmentEmbeddingExtractor:
    def __init__(self, cfg: EmbeddingFeatureConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if cfg.use_embedding:
            raise NotImplementedError(
                "Phase-1 does not enable embedding extraction by default. "
                "Integrate with src/models/audio if needed."
            )

    @property
    def out_dim(self) -> int:
        if self.model is None:
            return 0
        return int(getattr(self.model, "out_features"))

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
