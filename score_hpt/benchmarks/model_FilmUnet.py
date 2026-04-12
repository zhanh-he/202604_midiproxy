import logging
import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.velo_model.pretrained_utils import resolve_frontend_pretrained_checkpoint

# Inject a lightweight torchlibrosa shim so the original FiLM code can import it.
_VENDOR_ROOT = Path(__file__).resolve().parent
_AUDIO_SRC = _VENDOR_ROOT / "kim_ismir2024" / "src"
sys.path.insert(0, str(_AUDIO_SRC))
import audio_transforms  # type: ignore  # noqa: E402

torchlibrosa_mod = types.ModuleType("torchlibrosa")
torchlibrosa_stft_mod = types.ModuleType("torchlibrosa.stft")
torchlibrosa_stft_mod.Spectrogram = audio_transforms.Spectrogram
torchlibrosa_stft_mod.LogmelFilterBank = audio_transforms.LogmelFilterBank
sys.modules["torchlibrosa"] = torchlibrosa_mod
sys.modules["torchlibrosa.stft"] = torchlibrosa_stft_mod

# Reuse the untouched FiLM codebase.
_config_spec = importlib.util.spec_from_file_location(
    "kim_ismir2024_src_config",
    _AUDIO_SRC / "config.py",
)
kim_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(kim_config)

sys.modules["config"] = kim_config
_model_spec = importlib.util.spec_from_file_location(
    "kim_ismir2024_src_model",
    _AUDIO_SRC / "model.py",
)
_kim_model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_kim_model)
ScoreInformedMidiVelocityEstimator = _kim_model.ScoreInformedMidiVelocityEstimator


class FiLMUNet(nn.Module):
    """Thin wrapper around the original FiLM U-Net trained from scratch."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._set_conditioning()
        self._set_runtime_config(cfg)

        self.model = ScoreInformedMidiVelocityEstimator(
            frames_per_second=kim_config.frames_per_second,
            classes_num=kim_config.classes_num,
        )
        self._maybe_load_pretrained(cfg)

    @staticmethod
    def _set_conditioning() -> None:
        """FiLM U-Net in this repo always uses frame conditioning."""
        kim_config.condition_check = True
        kim_config.condition_type = "frame"

    @staticmethod
    def _set_runtime_config(cfg) -> None:
        kim_config.sample_rate = int(cfg.feature.sample_rate)
        kim_config.frames_per_second = int(cfg.feature.frames_per_second)
        kim_config.classes_num = int(cfg.feature.classes_num)

    def _maybe_load_pretrained(self, cfg) -> None:
        checkpoint_path = resolve_frontend_pretrained_checkpoint(
            getattr(cfg, "model", None),
            model_label="filmunet",
            required=False,
        )
        if checkpoint_path is None:
            return
        state_dict = self._prepare_state_dict(self._load_state_dict(checkpoint_path))
        self.model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
        keys = list(checkpoint.keys())
        if keys and all(k.startswith("module.") for k in keys):
            checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
        keys = list(checkpoint.keys())
        if keys and all(k.startswith("model.") for k in keys):
            checkpoint = {k.replace("model.", "", 1): v for k, v in checkpoint.items()}
        return checkpoint

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        prepared = {
            k: v
            for k, v in state_dict.items()
            if not (
                k.startswith("spectrogram_extractor")
                or k.startswith("logmel_extractor")
            )
        }
        local_state = self.model.state_dict()
        for key, tensor in local_state.items():
            if key.startswith("spectrogram_extractor") or key.startswith("logmel_extractor"):
                prepared[key] = tensor.detach().clone()
        return prepared

    @staticmethod
    def _align_score_time(score: Optional[torch.Tensor], target_frames: int) -> Optional[torch.Tensor]:
        if score is None or score.dim() != 3 or score.size(1) == target_frames:
            return score
        score_4d = score.unsqueeze(1)
        aligned = F.interpolate(
            score_4d,
            size=(target_frames, score.size(2)),
            mode="nearest",
        )
        return aligned[:, 0]

    def forward(self, waveform, score: Optional[torch.Tensor] = None):
        original_frames = None if score is None else int(score.size(1))
        if score is not None:
            hop_size = max(1, kim_config.sample_rate // kim_config.frames_per_second)
            target_frames = 1 + (waveform.shape[-1] // hop_size)
            score = self._align_score_time(score, target_frames)
        output = self.model(waveform, score)
        if original_frames is not None:
            velocity_output = output.get("velocity_output")
            if torch.is_tensor(velocity_output) and velocity_output.dim() >= 3:
                output = dict(output)
                output["velocity_output"] = velocity_output[:, :original_frames]
        return output


class FiLMUNetPretrained(FiLMUNet):
    """Compatibility alias of FiLMUNet."""
