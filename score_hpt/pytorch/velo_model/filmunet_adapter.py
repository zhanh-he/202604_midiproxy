from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from .base import BaseAdapter, ensure_btp88, register_adapter


class _BaseFiLMUNetAdapter(BaseAdapter):
    def forward(self, audio: torch.Tensor, score: Optional[torch.Tensor] = None, *args, **kwargs) -> Dict[str, Any]:
        out = self.model(audio, score)
        vel = out.get("velocity_output")
        if vel is None or not torch.is_tensor(vel):
            raise KeyError("FiLMUNetAdapter expected tensor output_dict['velocity_output'].")
        return self._finalize({"vel": ensure_btp88(vel, "vel"), "vel_logits": None, "extra": {}})


@register_adapter("filmunet")
class FiLMUNetAdapter(_BaseFiLMUNetAdapter):
    def __init__(self, model: Optional[nn.Module] = None, cfg=None):
        if model is None:
            if cfg is None:
                raise ValueError("FiLMUNetAdapter: cfg is required when model is None.")
            from benchmarks.model_FilmUnet import FiLMUNet

            model = FiLMUNet(cfg)
        super().__init__(model=model)


@register_adapter("filmunet_pretrained")
class FiLMUNetPretrainedAdapter(_BaseFiLMUNetAdapter):
    def __init__(self, model: Optional[nn.Module] = None, cfg=None):
        if model is None:
            if cfg is None:
                raise ValueError("FiLMUNetPretrainedAdapter: cfg is required when model is None.")
            from benchmarks.model_FilmUnet import FiLMUNetPretrained

            model = FiLMUNetPretrained(cfg)
        super().__init__(model=model)
