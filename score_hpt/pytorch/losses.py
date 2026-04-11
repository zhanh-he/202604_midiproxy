from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VELOCITY_SCALE = 128.0
DEFAULT_PIANO_SSM_FFT_BASE = (192, 384, 768, 1526, 3072, 6144, 12288)


# -----------------------------------------------------------------------------
# Velocity supervision losses
# -----------------------------------------------------------------------------


def _get_velocity_target(target_dict):
    """Get normalized velocity target with fixed scale for compatibility."""
    return target_dict["velocity_roll"] / VELOCITY_SCALE



def _get_velocity_pred(output_dict):
    """Fetch velocity prediction tensor, accepting both vel_corr and velocity_output keys."""
    if "vel_corr" in output_dict:
        return output_dict["vel_corr"]
    if "velocity_output" in output_dict:
        return output_dict["velocity_output"]
    raise KeyError("velocity prediction not found in output_dict (expected vel_corr or velocity_output)")



def has_supervised_velocity_target(target_dict):
    """Check whether a batch provides usable velocity supervision."""
    if target_dict is None:
        return False
    if "velocity_roll" not in target_dict:
        return False
    if target_dict["velocity_roll"] is None:
        return False
    if "has_velocity_target" in target_dict:
        flag = target_dict["has_velocity_target"]
        if torch.is_tensor(flag):
            return bool(torch.any(flag > 0))
        return bool(flag)
    return True



def _align_time_dim(*tensors):
    """Trim all tensors along time dimension to the minimum shared length."""
    if not tensors:
        return tensors
    min_steps = min(tensor.size(1) for tensor in tensors)
    if all(tensor.size(1) == min_steps for tensor in tensors):
        return tensors
    return tuple(tensor[:, :min_steps] for tensor in tensors)



def _masked_mean(values, mask):
    """Compute masked mean with time alignment and safe denominator."""
    values, mask = _align_time_dim(values, mask)
    mask = mask.to(values.dtype)
    denom = torch.sum(mask).clamp_min(1e-8)
    return torch.sum(values * mask) / denom



def _masked_bce(output, target, mask):
    """Binary crossentropy (BCE) with mask."""
    output, target, mask = _align_time_dim(output, target, mask)
    output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
    matrix = F.binary_cross_entropy(output, target, reduction="none")
    return _masked_mean(matrix, mask)



def _masked_mse(output, target, mask):
    """Mean squared error (MSE) with mask."""
    output, target, mask = _align_time_dim(output, target, mask)
    return _masked_mean((output - target) ** 2, mask)



def _masked_l1(output, target, mask):
    """Mean absolute error restricted by mask."""
    output, target, mask = _align_time_dim(output, target, mask)
    return _masked_mean(torch.abs(output - target), mask)


############ Velo Available - Velocity Supervised Loss ############


def _velocity_pointwise_loss(output_dict, target_dict, mask_key, pointwise_loss):
    pred = _get_velocity_pred(output_dict)
    target = _get_velocity_target(target_dict)
    return pointwise_loss(pred, target, target_dict[mask_key])



def velocity_bce(cfg, output_dict, target_dict, cond_dict=None):
    """Velocity regression loss only, using BCE as in HPT."""
    return _velocity_pointwise_loss(output_dict, target_dict, "onset_roll", _masked_bce)



def velocity_mse(cfg, output_dict, target_dict, cond_dict=None):
    """Velocity regression loss only, using MSE as in ONF."""
    return _velocity_pointwise_loss(output_dict, target_dict, "onset_roll", _masked_mse)



def kim_velocity_bce_l1(cfg, output_dict, target_dict, cond_dict=None):
    """BCE + L1 hybrid loss proposed by Kim et al. (ISMIR 2024)."""
    theta = cfg.loss.kim_loss_alpha
    pred = _get_velocity_pred(output_dict)
    onset_target = _get_velocity_target(target_dict)
    bce_loss = _masked_bce(pred, onset_target, target_dict["frame_roll"])
    l1_loss = _masked_l1(pred, onset_target, target_dict["onset_roll"])
    return theta * bce_loss + (1 - theta) * l1_loss


############ Velo Unavailable - Audio Supervised Prior ############


def velocity_prior_loss(cfg, output_dict, target_dict=None):
    """
    Weak anti-collapse prior for proxy-only or proxy-heavy training.

    L = (mean(v) - mu)^2 + relu(var_min - var(v))
    By default the prior is evaluated on onset positions only.
    """
    weight = float(getattr(cfg.loss, "velocity_prior_weight", 0.0) or 0.0)
    pred = _get_velocity_pred(output_dict)
    if weight <= 0:
        return pred.new_tensor(0.0)

    use_onsets_only = bool(getattr(cfg.loss, "velocity_prior_onset_only", True))
    mask = None
    if use_onsets_only and target_dict is not None and "onset_roll" in target_dict:
        mask = target_dict["onset_roll"]
        pred, mask = _align_time_dim(pred, mask)
        values = pred[mask > 0]
    else:
        values = pred.reshape(-1)

    if values.numel() == 0:
        return pred.new_tensor(0.0)

    prior_mean = float(getattr(cfg.loss, "velocity_prior_mean", 0.5))
    prior_min_var = float(getattr(cfg.loss, "velocity_prior_min_var", 0.01))
    mean_term = (values.mean() - prior_mean) ** 2
    var_term = F.relu(pred.new_tensor(prior_min_var) - values.var(unbiased=False))
    return mean_term + var_term


# -----------------------------------------------------------------------------
# Audio supervision losses for differentiable proxy training
# -----------------------------------------------------------------------------


def _maybe_import_librosa():
    try:
        import librosa
    except Exception as exc:  # pragma: no cover - dependency error path
        raise ImportError(
            "librosa is required to build mel/chroma/loudness filterbanks for Piano-SSM and DDSP audio losses."
        ) from exc
    return librosa



def _maybe_import_librosa_filters():
    return _maybe_import_librosa().filters



def _mean_difference(
    target: torch.Tensor,
    value: torch.Tensor,
    loss_type: str = "L1",
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Common reduction used by DDSP-style losses."""
    difference = target - value
    weights = 1.0 if weights is None else weights
    loss_type = str(loss_type).upper()

    if loss_type == "L1":
        return torch.mean(torch.abs(difference * weights))
    if loss_type == "L2":
        return torch.mean((difference ** 2) * weights)
    if loss_type == "COSINE":
        target_flat = target.reshape(target.size(0), -1)
        value_flat = value.reshape(value.size(0), -1)
        cosine_loss = 1.0 - F.cosine_similarity(target_flat, value_flat, dim=-1)
        if torch.is_tensor(weights):
            weights = weights.reshape(weights.size(0), -1).mean(dim=-1)
        return torch.mean(cosine_loss * weights)
    raise ValueError(f"Invalid loss_type: {loss_type}")



def _canonicalize_audio(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio to mono batch-major shape [B, samples]."""
    if not torch.is_tensor(audio):
        raise TypeError(f"Expected a torch.Tensor audio input, got {type(audio)!r}")

    if audio.dim() == 1:
        return audio.unsqueeze(0)
    if audio.dim() == 2:
        return audio
    if audio.dim() != 3:
        raise ValueError(f"Expected audio tensor with 1, 2, or 3 dims, got shape {tuple(audio.shape)}")

    if audio.size(1) == 1:
        return audio.squeeze(1)
    if audio.size(-1) == 1:
        return audio.squeeze(-1)

    # Graceful fallback for small multi-channel tensors.
    if audio.size(1) <= 4:
        return audio.mean(dim=1)
    if audio.size(-1) <= 4:
        return audio.mean(dim=-1)

    raise ValueError(f"Cannot infer mono axis for audio tensor with shape {tuple(audio.shape)}")



def _align_audio_pair(pred_audio: torch.Tensor, target_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert audio tensors to [B, S] and trim to the shared minimum length."""
    pred_audio = _canonicalize_audio(pred_audio)
    target_audio = _canonicalize_audio(target_audio)
    min_len = min(pred_audio.size(-1), target_audio.size(-1))
    pred_audio = pred_audio[..., :min_len]
    target_audio = target_audio[..., :min_len]
    return pred_audio, target_audio



def _get_audio_loss_cfg(cfg, section_name: Optional[str] = None):
    audio_cfg = getattr(getattr(cfg, "proxy", None), "audio_loss", None)
    if audio_cfg is None:
        return None
    if section_name is None:
        return audio_cfg
    nested = getattr(audio_cfg, section_name, None)
    return nested if nested is not None else audio_cfg



def _piano_ssm_default_fft_sizes(sample_rate: int) -> Tuple[int, ...]:
    """Replicate the Piano-SSM repo default scaling for SpectralLoss."""
    sr = max(1, int(sample_rate))
    values = [max(2, int(sr / 48000.0 * base)) for base in DEFAULT_PIANO_SSM_FFT_BASE]
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)



def get_audio_loss_name(cfg) -> str:
    """Return the canonical proxy audio-loss name from the config."""
    audio_cfg = getattr(getattr(cfg, "proxy", None), "audio_loss", None)
    raw_name = None
    if audio_cfg is not None:
        raw_name = getattr(audio_cfg, "type", None)
        if raw_name is None:
            raw_name = getattr(audio_cfg, "name", None)
    raw_name = str(raw_name or "piano_ssm_spectral_plus_log_rms").strip().lower()

    alias_map = {
        "combinedspectralloss": "piano_ssm_combined",
        "combined_spectral_loss": "piano_ssm_combined",
        "combined_spectral": "piano_ssm_combined",
        "smml": "piano_ssm_combined",
        "piano_ssm_combinedspectralloss": "piano_ssm_combined",
        "piano_ssm_combined_spectral_loss": "piano_ssm_combined",
        "spectralloss": "piano_ssm_spectral",
        "spectral_loss": "piano_ssm_spectral",
        "mssl": "piano_ssm_spectral",
        "piano_ssm_spectralloss": "piano_ssm_spectral",
        "piano_ssm_spectral_loss": "piano_ssm_spectral",
        "spectral_plus_log_rms": "piano_ssm_spectral_plus_log_rms",
        "piano_ssm_spectral_plus_logrms": "piano_ssm_spectral_plus_log_rms",
        "piano_ssm_spectral_plus_rms": "piano_ssm_spectral_plus_log_rms",
        "spectral_plus_loudness": "piano_ssm_spectral_plus_ddsp_loudness",
        "piano_ssm_spectral_plus_loudness": "piano_ssm_spectral_plus_ddsp_loudness",
        "piano_ssm_spectral_plus_ddsploudness": "piano_ssm_spectral_plus_ddsp_loudness",
        "combined_rm": "piano_ssm_combined_rm",
        "piano_ssm_combinedrm": "piano_ssm_combined_rm",
        "loudness": "ddsp_piano_loudness",
        "ddsp_loudness": "ddsp_piano_loudness",
        "ddsp_piano_loudnessloss": "ddsp_piano_loudness",
    }
    return alias_map.get(raw_name, raw_name)


class _BaseAudioLoss(nn.Module):
    """Shared utilities for proxy audio losses."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = float(eps)
        self._window_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}
        self._filter_cache: Dict[Tuple[str, int, int, int, str, str], torch.Tensor] = {}

    def _window(self, win_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(win_length), str(device), str(dtype))
        if key not in self._window_cache:
            self._window_cache[key] = torch.hann_window(int(win_length), device=device, dtype=dtype)
        return self._window_cache[key]

    def _filter_bank(
        self,
        kind: str,
        n_fft: int,
        n_bins: int,
        sample_rate: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (kind, int(n_fft), int(n_bins), int(sample_rate), str(device), str(dtype))
        if key in self._filter_cache:
            return self._filter_cache[key]

        librosa_filters = _maybe_import_librosa_filters()
        if kind == "mel":
            fb_np = librosa_filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_bins)
        elif kind == "chroma":
            fb_np = librosa_filters.chroma(sr=sample_rate, n_fft=n_fft, n_chroma=n_bins)
        else:
            raise ValueError(f"Unsupported filter-bank kind: {kind}")

        fb = torch.tensor(fb_np, device=device, dtype=dtype)
        self._filter_cache[key] = fb
        return fb

    def _stft_mag(
        self,
        audio: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "reflect",
    ) -> torch.Tensor:
        audio = _canonicalize_audio(audio)
        n_fft = max(2, int(n_fft))
        hop_length = max(1, int(hop_length))
        win_length = n_fft if win_length is None else max(2, int(win_length))
        window = self._window(win_length, audio.device, audio.dtype)
        effective_pad_mode = pad_mode
        if center and pad_mode != "constant":
            half_n_fft = n_fft // 2
            if audio.size(-1) <= half_n_fft:
                effective_pad_mode = "constant"
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=effective_pad_mode,
            return_complex=True,
        )
        return spec.abs().clamp_min(self.eps)

    def _audio_stats(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "audio_rms_pred": pred_audio.pow(2).mean().sqrt().detach(),
            "audio_rms_target": target_audio.pow(2).mean().sqrt().detach(),
        }


class GlobalRMSLoss(_BaseAudioLoss):
    """
    Clip-level RMS matching.

    This is the coarsest loudness loss in the file.
    It only compares one energy value per audio clip.
    It is useful as a low-cost auxiliary term.
    """

    def __init__(self, loss_type: str = "L1", eps: float = 1e-7) -> None:
        super().__init__(eps=eps)
        self.loss_type = str(loss_type).upper()

    def _rms(self, audio: torch.Tensor) -> torch.Tensor:
        return audio.pow(2).mean(dim=-1).clamp_min(self.eps).sqrt()

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        pred_rms = self._rms(pred_audio)
        target_rms = self._rms(target_audio)
        loss = _mean_difference(target_rms, pred_rms, loss_type=self.loss_type)

        stats = {
            "rms_loss": loss.detach(),
            "rms_pred": pred_rms.mean().detach(),
            "rms_target": target_rms.mean().detach(),
        }
        stats.update(self._audio_stats(pred_audio, target_audio))
        return loss, stats


class GlobalLogRMSLoss(_BaseAudioLoss):
    """
    Clip-level loudness loss on log-RMS or dB-RMS.

    Compared with plain RMS, log-RMS is less dominated by large peaks.
    This makes it a better auxiliary term when the target variable is related
    to perceived dynamics instead of raw waveform scale.
    """

    def __init__(
        self,
        loss_type: str = "L1",
        db_scale: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(eps=eps)
        self.loss_type = str(loss_type).upper()
        self.db_scale = bool(db_scale)

    def _log_rms(self, audio: torch.Tensor) -> torch.Tensor:
        rms = audio.pow(2).mean(dim=-1).clamp_min(self.eps).sqrt()
        if self.db_scale:
            return 20.0 * torch.log10(rms)
        return torch.log(rms)

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        pred_log_rms = self._log_rms(pred_audio)
        target_log_rms = self._log_rms(target_audio)
        loss = _mean_difference(target_log_rms, pred_log_rms, loss_type=self.loss_type)

        stats = {
            "log_rms_loss": loss.detach(),
            "log_rms_pred": pred_log_rms.mean().detach(),
            "log_rms_target": target_log_rms.mean().detach(),
        }
        stats.update(self._audio_stats(pred_audio, target_audio))
        return loss, stats


class DDSPPianoLoudnessLoss(_BaseAudioLoss):
    """
    DDSP-style perceptual loudness loss.

    This is a PyTorch implementation of the loudness term used by DDSP.
    It computes an A-weighted power loudness curve in dB and compares the
    frame-wise trajectories. This is much closer to `ddsp.losses.SpectralLoss`
    with `loudness_weight > 0` than a plain clip-level RMS loss.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_rate: int = 250,
        n_fft: int = 2048,
        range_db: float = 120.0,
        ref_db: float = 0.0,
        loss_type: str = "L1",
        eps: float = 1e-7,
    ) -> None:
        super().__init__(eps=eps)
        self.sample_rate = int(sample_rate)
        self.frame_rate = int(frame_rate)
        self.n_fft = int(n_fft)
        self.range_db = float(range_db)
        self.ref_db = float(ref_db)
        self.loss_type = str(loss_type).upper()
        self._a_weight_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = {}

    def _a_weighting(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (self.sample_rate, self.n_fft, str(device), str(dtype))
        if key in self._a_weight_cache:
            return self._a_weight_cache[key]

        librosa = _maybe_import_librosa()
        frequencies = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        frequencies = np.maximum(frequencies, 1e-7)
        a_weighting_db = librosa.A_weighting(frequencies)
        a_weighting_db = np.nan_to_num(a_weighting_db, neginf=-80.0, posinf=80.0).astype(np.float32)

        a_weighting = torch.tensor(a_weighting_db, device=device, dtype=dtype)
        weighting = torch.pow(a_weighting.new_tensor(10.0), a_weighting / 10.0).view(1, -1, 1)
        self._a_weight_cache[key] = weighting
        return weighting

    def _loudness_curve(self, audio: torch.Tensor) -> torch.Tensor:
        audio = _canonicalize_audio(audio)
        hop_length = max(1, int(self.sample_rate // self.frame_rate))
        window = self._window(self.n_fft, audio.device, audio.dtype)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        power = spec.abs().pow(2)
        power = power * self._a_weighting(audio.device, audio.dtype)
        avg_power = power.mean(dim=1)
        loudness = 10.0 * torch.log10(avg_power.clamp_min(self.eps)) - self.ref_db
        loudness = torch.clamp(loudness, min=-self.range_db)
        return loudness

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        pred_loudness = self._loudness_curve(pred_audio)
        target_loudness = self._loudness_curve(target_audio)
        loss = _mean_difference(target_loudness, pred_loudness, loss_type=self.loss_type, weights=weights)

        stats = {
            "loudness_loss": loss.detach(),
            "loudness_pred_mean": pred_loudness.mean().detach(),
            "loudness_target_mean": target_loudness.mean().detach(),
        }
        stats.update(self._audio_stats(pred_audio, target_audio))
        return loss, stats


class CompositeAudioLoss(nn.Module):
    """Weighted sum of named audio losses."""

    def __init__(self, named_losses):
        super().__init__()
        self.loss_modules = nn.ModuleDict({name: module for name, module, _ in named_losses})
        self.loss_weights = {name: float(weight) for name, _, weight in named_losses}

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        total_loss = pred_audio.new_tensor(0.0)
        stats: Dict[str, torch.Tensor] = {}

        for name, module in self.loss_modules.items():
            weight = self.loss_weights[name]
            if weight <= 0:
                continue
            sub_loss, sub_stats = module(pred_audio, target_audio)
            weighted_loss = weight * sub_loss
            total_loss = total_loss + weighted_loss
            stats[f"{name}_raw"] = sub_loss.detach()
            stats[f"{name}_weighted"] = weighted_loss.detach()
            for key, value in sub_stats.items():
                if torch.is_tensor(value):
                    stats[f"{name}_{key}"] = value.detach()
                else:
                    stats[f"{name}_{key}"] = value

        return total_loss, stats


class PianoSSMSpectralPlusLogRMSLoss(CompositeAudioLoss):
    """
    Conservative proxy loss for velocity experiments.

    Main term: Piano-SSM spectral loss.
    Auxiliary term: clip-level log-RMS loudness.

    This keeps the strong spectral supervision and adds a small loudness bias.
    """

    def __init__(
        self,
        spectral_loss: PianoSSMSpectralLoss,
        log_rms_loss: GlobalLogRMSLoss,
        spectral_weight: float = 1.0,
        log_rms_weight: float = 0.05,
    ) -> None:
        super().__init__([
            ("spectral", spectral_loss, spectral_weight),
            ("log_rms", log_rms_loss, log_rms_weight),
        ])


class PianoSSMSpectralPlusDDSPLoudnessLoss(CompositeAudioLoss):
    """
    Piano-SSM spectral loss with DDSP-style loudness-curve auxiliary.

    This variant is stronger than the clip-level log-RMS version because it
    matches a frame-wise loudness contour instead of one scalar per clip.
    """

    def __init__(
        self,
        spectral_loss: PianoSSMSpectralLoss,
        loudness_loss: DDSPPianoLoudnessLoss,
        spectral_weight: float = 1.0,
        loudness_weight: float = 0.05,
    ) -> None:
        super().__init__([
            ("spectral", spectral_loss, spectral_weight),
            ("ddsp_loudness", loudness_loss, loudness_weight),
        ])


class MultiScaleSTFTLoss(_BaseAudioLoss):
    """Simple multi-scale spectral loss for proxy audio matching."""

    def __init__(
        self,
        fft_sizes: Iterable[int] = (512, 1024, 2048),
        hop_ratio: float = 0.25,
        win_ratio: float = 1.0,
        spectral_convergence_weight: float = 1.0,
        log_mag_weight: float = 1.0,
        lin_mag_weight: float = 0.0,
        waveform_l1_weight: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(eps=eps)
        self.fft_sizes = tuple(int(v) for v in fft_sizes)
        self.hop_ratio = float(hop_ratio)
        self.win_ratio = float(win_ratio)
        self.spectral_convergence_weight = float(spectral_convergence_weight)
        self.log_mag_weight = float(log_mag_weight)
        self.lin_mag_weight = float(lin_mag_weight)
        self.waveform_l1_weight = float(waveform_l1_weight)

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)

        total_loss = pred_audio.new_tensor(0.0)
        stats: Dict[str, torch.Tensor] = {}

        if self.waveform_l1_weight > 0:
            wave_l1 = torch.mean(torch.abs(pred_audio - target_audio))
            total_loss = total_loss + self.waveform_l1_weight * wave_l1
            stats["waveform_l1"] = wave_l1.detach()

        sc_terms = []
        log_terms = []
        lin_terms = []
        for n_fft in self.fft_sizes:
            hop_length = max(1, int(round(n_fft * self.hop_ratio)))
            win_length = max(2, int(round(n_fft * self.win_ratio)))
            pred_mag = self._stft_mag(pred_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            target_mag = self._stft_mag(target_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

            flat_diff = (target_mag - pred_mag).flatten(1)
            flat_target = target_mag.flatten(1)
            sc = (flat_diff.norm(dim=1) / (flat_target.norm(dim=1) + self.eps)).mean()
            log_mag = torch.mean(torch.abs(torch.log(target_mag) - torch.log(pred_mag)))
            lin_mag = torch.mean(torch.abs(target_mag - pred_mag))

            sc_terms.append(sc)
            log_terms.append(log_mag)
            lin_terms.append(lin_mag)

            total_loss = total_loss + self.spectral_convergence_weight * sc
            total_loss = total_loss + self.log_mag_weight * log_mag
            total_loss = total_loss + self.lin_mag_weight * lin_mag

        if sc_terms:
            stats["stft_sc"] = torch.stack(sc_terms).mean().detach()
            stats["stft_log"] = torch.stack(log_terms).mean().detach()
            stats["stft_lin"] = torch.stack(lin_terms).mean().detach()

        stats.update(self._audio_stats(pred_audio, target_audio))
        return total_loss, stats


class _ScaledSTFTLoss(_BaseAudioLoss):
    """A small differentiable STFT loss with optional mel/chroma scaling."""

    def __init__(
        self,
        sample_rate: int,
        fft_size: int,
        hop_size: int,
        win_length: int,
        scale: Optional[str] = None,
        n_bins: Optional[int] = None,
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        scale_invariance: bool = False,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(eps=eps)
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.win_length = int(win_length)
        self.scale = None if scale is None else str(scale).strip().lower()
        self.n_bins = None if n_bins is None else int(n_bins)
        self.w_sc = float(w_sc)
        self.w_log_mag = float(w_log_mag)
        self.w_lin_mag = float(w_lin_mag)
        self.scale_invariance = bool(scale_invariance)

        if self.scale is not None and self.n_bins is None:
            raise ValueError("n_bins must be provided when using a scaled STFT loss")

    def _scaled_mag(self, audio: torch.Tensor) -> torch.Tensor:
        mag = self._stft_mag(
            audio,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            center=True,
            pad_mode="reflect",
        )
        if self.scale is None:
            return mag

        fb = self._filter_bank(
            kind=self.scale,
            n_fft=self.fft_size,
            n_bins=self.n_bins,
            sample_rate=self.sample_rate,
            device=mag.device,
            dtype=mag.dtype,
        )
        return torch.matmul(fb, mag).clamp_min(self.eps)

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        return_components: bool = False,
    ):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        pred_mag = self._scaled_mag(pred_audio)
        target_mag = self._scaled_mag(target_audio)

        if self.scale_invariance:
            alpha = (pred_mag * target_mag).sum(dim=(-2, -1)) / (target_mag.pow(2).sum(dim=(-2, -1)) + self.eps)
            target_mag = target_mag * alpha.view(-1, 1, 1)

        sc = pred_mag.new_tensor(0.0)
        log_mag = pred_mag.new_tensor(0.0)
        lin_mag = pred_mag.new_tensor(0.0)

        if self.w_sc > 0:
            sc = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + self.eps)
        if self.w_log_mag > 0:
            log_mag = torch.mean(torch.abs(torch.log(target_mag) - torch.log(pred_mag)))
        if self.w_lin_mag > 0:
            lin_mag = torch.mean(torch.abs(target_mag - pred_mag))

        total = (self.w_sc * sc) + (self.w_log_mag * log_mag) + (self.w_lin_mag * lin_mag)

        if return_components:
            return total, {
                "sc": sc.detach(),
                "log_mag": log_mag.detach(),
                "lin_mag": lin_mag.detach(),
            }
        return total


class PianoSSMSpectralLoss(_BaseAudioLoss):
    """
    Piano-SSM / DDSP-style spectral loss.

    This follows the Piano-SSM repository implementation of SpectralLoss and keeps
    the overlap-based STFT padding behaviour intact.
    """

    def __init__(
        self,
        fft_sizes: Optional[Iterable[int]] = None,
        loss_type: str = "L1",
        mag_weight: float = 1.0,
        logmag_weight: float = 0.0,
        overlap: float = 0.75,
        sample_rate: Optional[int] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(eps=eps)
        if fft_sizes is None:
            if sample_rate is None:
                fft_sizes = (2048, 1024, 512, 256, 128, 64)
            else:
                fft_sizes = _piano_ssm_default_fft_sizes(int(sample_rate))
        self.fft_sizes = tuple(int(size) for size in fft_sizes)
        self.loss_type = str(loss_type).upper()
        self.mag_weight = float(mag_weight)
        self.logmag_weight = float(logmag_weight)
        self.overlap = float(overlap)

    def compute_mag(self, audio: torch.Tensor, size: int) -> torch.Tensor:
        """Compute the repo-style magnitude STFT with explicit tail padding."""
        audio = _canonicalize_audio(audio)
        frame_length = max(2, int(size))
        frame_step = max(1, int(round(frame_length * (1.0 - self.overlap))))
        window = self._window(frame_length, audio.device, audio.dtype)

        total_frames = max(1, (audio.size(-1) + frame_step - 1) // frame_step)
        padded_length = (total_frames - 1) * frame_step + frame_length
        pad_size = max(0, padded_length - audio.size(-1))
        audio = F.pad(audio, (0, pad_size))

        stft_output = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=frame_step,
            win_length=frame_length,
            window=window,
            center=False,
            pad_mode="constant",
            return_complex=True,
        )
        return stft_output.abs().permute(0, 2, 1).clamp_min(self.eps)

    def mean_difference(
        self,
        target: torch.Tensor,
        value: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _mean_difference(target, value, loss_type=self.loss_type, weights=weights)

    def safe_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        epoch: int = 0,
        weights: Optional[torch.Tensor] = None,
    ):
        del epoch  # kept for interface compatibility
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)

        total_loss = pred_audio.new_tensor(0.0)
        stats: Dict[str, torch.Tensor] = {}

        for size in self.fft_sizes:
            target_mag = self.compute_mag(target_audio, size)
            pred_mag = self.compute_mag(pred_audio, size)

            if self.mag_weight > 0:
                loss_mag = self.mag_weight * self.mean_difference(target_mag, pred_mag, weights=weights)
                total_loss = total_loss + loss_mag
                stats[f"mag_{size}"] = loss_mag.detach()

            if self.logmag_weight > 0:
                target_logmag = self.safe_log(target_mag)
                pred_logmag = self.safe_log(pred_mag)
                loss_logmag = self.logmag_weight * self.mean_difference(target_logmag, pred_logmag, weights=weights)
                total_loss = total_loss + loss_logmag
                stats[f"logmag_{size}"] = loss_logmag.detach()

        stats.update(self._audio_stats(pred_audio, target_audio))
        return total_loss, stats


class PianoSSMCombinedSpectralLoss(_BaseAudioLoss):
    """
    Piano-SSM STFT-Mel-Mean loss (SMML).

    The implementation mirrors the public Piano-SSM repository behaviour while
    avoiding a hard dependency on the `auraloss` package.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: Optional[int] = None,
        stride: Optional[int] = None,
        mel_fft_size: Optional[int] = None,
        mel_win_length: Optional[int] = None,
        mel_hop_size: Optional[int] = None,
        mel_n_mels: int = 128,
        mean_loss_abs: bool = False,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(eps=eps)
        self.sample_rate = int(sample_rate)
        self.frame_length = int(frame_length or self.sample_rate)
        self.stride = int(stride or max(1, self.frame_length // 10))
        self.mean_loss_abs = bool(mean_loss_abs)

        # Match the scaling used in the Piano-SSM repo.
        scale = max(float(self.sample_rate) / 44100.0, 1e-8)
        mel_fft_size = int(mel_fft_size or max(2, int((1024 * 2) // scale)))
        mel_win_length = int(mel_win_length or max(2, int(1024 // scale)))
        mel_hop_size = int(mel_hop_size or max(1, int(256 // scale)))

        self.mel_loss_module = _ScaledSTFTLoss(
            sample_rate=self.sample_rate,
            fft_size=mel_fft_size,
            hop_size=mel_hop_size,
            win_length=mel_win_length,
            scale="mel",
            n_bins=int(mel_n_mels),
            w_sc=1.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
            scale_invariance=False,
            eps=eps,
        )

    def fft_loss(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        pred_mag = self._stft_mag(
            pred_audio,
            n_fft=self.frame_length,
            hop_length=self.stride,
            win_length=self.frame_length,
            center=True,
            pad_mode="reflect",
        )
        target_mag = self._stft_mag(
            target_audio,
            n_fft=self.frame_length,
            hop_length=self.stride,
            win_length=self.frame_length,
            center=True,
            pad_mode="reflect",
        )
        return torch.mean(torch.abs(target_mag - pred_mag))

    def mean_loss(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        # Keep the original Piano-SSM formulation by default.
        mean_term = torch.mean(torch.mean(pred_audio, dim=1).pow(2) - torch.mean(target_audio, dim=1).pow(2))
        if self.mean_loss_abs:
            mean_term = torch.abs(mean_term)
        return mean_term

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        diff_abs = self.fft_loss(pred_audio, target_audio)
        mel_loss, mel_stats = self.mel_loss_module(pred_audio, target_audio, return_components=True)
        mean_loss = self.mean_loss(pred_audio, target_audio)

        total_loss = diff_abs + mel_loss + mean_loss
        stats = {
            "diff_abs": diff_abs.detach(),
            "mel_loss": mel_loss.detach(),
            "mel_sc": mel_stats["sc"],
            "mel_log_mag": mel_stats["log_mag"],
            "mel_lin_mag": mel_stats["lin_mag"],
            "mean_loss": mean_loss.detach(),
        }
        stats.update(self._audio_stats(pred_audio, target_audio))
        return total_loss, stats


class PianoSSMCombinedRMLoss(PianoSSMCombinedSpectralLoss):
    """
    Piano-SSM STFT-Mel-RM loss.

    RM means that the original Piano-SSM `mean_loss` term is replaced by a
    loudness-aware RMS term. The default uses log-RMS in dB.

    Old SMML tail term:
        mean(mean(pred)^2 - mean(target)^2)

    New RM tail term:
        global log-RMS loss or global RMS loss

    The new term is always non-negative and has a much clearer loudness
    interpretation. This is better aligned with velocity modelling.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_length: Optional[int] = None,
        stride: Optional[int] = None,
        mel_fft_size: Optional[int] = None,
        mel_win_length: Optional[int] = None,
        mel_hop_size: Optional[int] = None,
        mel_n_mels: int = 128,
        rm_mode: str = "log_rms",
        rm_weight: float = 1.0,
        rm_loss_type: str = "L1",
        rm_db_scale: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            frame_length=frame_length,
            stride=stride,
            mel_fft_size=mel_fft_size,
            mel_win_length=mel_win_length,
            mel_hop_size=mel_hop_size,
            mel_n_mels=mel_n_mels,
            mean_loss_abs=False,
            eps=eps,
        )
        self.rm_mode = str(rm_mode).strip().lower()
        self.rm_weight = float(rm_weight)
        if self.rm_mode == "rms":
            self.rm_loss_module = GlobalRMSLoss(loss_type=rm_loss_type, eps=eps)
        elif self.rm_mode in ("log_rms", "logrms", "db_rms", "db"):
            self.rm_loss_module = GlobalLogRMSLoss(
                loss_type=rm_loss_type,
                db_scale=rm_db_scale,
                eps=eps,
            )
        else:
            raise ValueError(f"Unsupported rm_mode: {rm_mode}")

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        diff_abs = self.fft_loss(pred_audio, target_audio)
        mel_loss, mel_stats = self.mel_loss_module(pred_audio, target_audio, return_components=True)
        rm_loss, rm_stats = self.rm_loss_module(pred_audio, target_audio)

        total_loss = diff_abs + mel_loss + self.rm_weight * rm_loss
        stats = {
            "diff_abs": diff_abs.detach(),
            "mel_loss": mel_loss.detach(),
            "mel_sc": mel_stats["sc"],
            "mel_log_mag": mel_stats["log_mag"],
            "mel_lin_mag": mel_stats["lin_mag"],
            "rm_loss": rm_loss.detach(),
            "rm_weighted": (self.rm_weight * rm_loss).detach(),
        }
        for key, value in rm_stats.items():
            if key in {"audio_rms_pred", "audio_rms_target"}:
                continue
            stats[f"rm_{key}"] = value
        stats.update(self._audio_stats(pred_audio, target_audio))
        return total_loss, stats


class PianoSSMChromaLoss(_BaseAudioLoss):
    """
    Differentiable chroma-domain proxy loss.

    The public Piano-SSM repository includes a `ChromaLoss` utility based on
    `librosa.feature.chroma_cqt`. Here we implement a differentiable STFT-chroma
    approximation so that gradients can still flow back through the proxy.
    """

    def __init__(
        self,
        sample_rate: int,
        fft_size: int = 4096,
        hop_size: int = 512,
        win_length: Optional[int] = None,
        n_chroma: int = 12,
        threshold: float = 0.3,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(eps=eps)
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.win_length = int(win_length or fft_size)
        self.n_chroma = int(n_chroma)
        self.threshold = float(threshold)

    def _chroma(self, audio: torch.Tensor) -> torch.Tensor:
        mag = self._stft_mag(
            audio,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            center=True,
            pad_mode="reflect",
        )
        fb = self._filter_bank(
            kind="chroma",
            n_fft=self.fft_size,
            n_bins=self.n_chroma,
            sample_rate=self.sample_rate,
            device=mag.device,
            dtype=mag.dtype,
        )
        chroma = torch.matmul(fb, mag).clamp_min(self.eps)
        chroma = chroma / chroma.amax(dim=1, keepdim=True).clamp_min(self.eps)
        return chroma.transpose(1, 2)

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        ref_chroma = self._chroma(target_audio)
        pred_chroma = self._chroma(pred_audio)

        mask = (ref_chroma > self.threshold).to(pred_chroma.dtype)
        tar = ref_chroma.clamp(0.0, 1.0)
        gen = pred_chroma.clamp(self.eps, 1.0 - self.eps)
        denom = mask.sum().clamp_min(1.0)
        loss = torch.sum(torch.abs(tar - gen) * mask) / denom

        stats = {"chroma_loss": loss.detach()}
        stats.update(self._audio_stats(pred_audio, target_audio))
        return loss, stats


AUDIO_LOSS_TYPES = {
    "piano_ssm_combined": PianoSSMCombinedSpectralLoss,
    "piano_ssm_combined_rm": PianoSSMCombinedRMLoss,
    "piano_ssm_spectral": PianoSSMSpectralLoss,
    "piano_ssm_spectral_plus_log_rms": PianoSSMSpectralPlusLogRMSLoss,
    "piano_ssm_spectral_plus_ddsp_loudness": PianoSSMSpectralPlusDDSPLoudnessLoss,
    "ddsp_piano_loudness": DDSPPianoLoudnessLoss,
}



def _build_piano_ssm_spectral_loss_from_cfg(cfg, sample_rate: int) -> PianoSSMSpectralLoss:
    audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_spectral")
    fft_sizes = getattr(audio_cfg, "fft_sizes", None)
    if fft_sizes is not None:
        fft_sizes = tuple(int(v) for v in fft_sizes)
        if len(fft_sizes) == 0:
            fft_sizes = None
    return PianoSSMSpectralLoss(
        fft_sizes=fft_sizes,
        loss_type=str(getattr(audio_cfg, "loss_type", "L1")),
        mag_weight=float(getattr(audio_cfg, "mag_weight", 1.0)),
        logmag_weight=float(getattr(audio_cfg, "logmag_weight", 1.0)),
        overlap=float(getattr(audio_cfg, "overlap", 0.75)),
        sample_rate=sample_rate,
        eps=float(getattr(audio_cfg, "eps", 1e-5)),
    )


def _build_log_rms_aux_loss_from_cfg(cfg) -> GlobalLogRMSLoss:
    audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_spectral_plus_log_rms")
    return GlobalLogRMSLoss(
        loss_type=str(getattr(audio_cfg, "loss_type", "L1")),
        db_scale=bool(getattr(audio_cfg, "db_scale", True)),
        eps=float(getattr(audio_cfg, "eps", 1e-7)),
    )



def _build_ddsp_piano_loudness_loss_from_cfg(cfg, sample_rate: int, frame_rate: int | None = None) -> DDSPPianoLoudnessLoss:
    audio_cfg = _get_audio_loss_cfg(cfg, "ddsp_piano_loudness")
    return DDSPPianoLoudnessLoss(
        sample_rate=sample_rate,
        frame_rate=int(frame_rate if frame_rate is not None else getattr(audio_cfg, "frame_rate", 250)),
        n_fft=int(getattr(audio_cfg, "n_fft", 2048)),
        range_db=float(getattr(audio_cfg, "range_db", 120.0)),
        ref_db=float(getattr(audio_cfg, "ref_db", 0.0)),
        loss_type=str(getattr(audio_cfg, "loss_type", "L1")),
        eps=float(getattr(audio_cfg, "eps", 1e-7)),
    )


def build_audio_loss(cfg, sample_rate_override=None, frame_rate_override=None):
    """Build the proxy audio loss selected by `proxy.audio_loss.type`."""
    audio_loss_name = get_audio_loss_name(cfg)
    if sample_rate_override is not None:
        sample_rate = int(sample_rate_override)
    else:
        sample_rate = int(
            getattr(getattr(cfg, "proxy", None), "sample_rate", getattr(getattr(cfg, "feature", None), "sample_rate", 16000))
        )

    if audio_loss_name == "piano_ssm_combined":
        audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_combined")
        return PianoSSMCombinedSpectralLoss(
            sample_rate=sample_rate,
            frame_length=getattr(audio_cfg, "frame_length", None),
            stride=getattr(audio_cfg, "stride", None),
            mel_fft_size=getattr(audio_cfg, "mel_fft_size", None),
            mel_win_length=getattr(audio_cfg, "mel_win_length", None),
            mel_hop_size=getattr(audio_cfg, "mel_hop_size", None),
            mel_n_mels=int(getattr(audio_cfg, "mel_n_mels", 128)),
            mean_loss_abs=bool(getattr(audio_cfg, "mean_loss_abs", False)),
            eps=float(getattr(audio_cfg, "eps", 1e-7)),
        )

    if audio_loss_name == "piano_ssm_combined_rm":
        audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_combined_rm")
        return PianoSSMCombinedRMLoss(
            sample_rate=sample_rate,
            frame_length=getattr(audio_cfg, "frame_length", None),
            stride=getattr(audio_cfg, "stride", None),
            mel_fft_size=getattr(audio_cfg, "mel_fft_size", None),
            mel_win_length=getattr(audio_cfg, "mel_win_length", None),
            mel_hop_size=getattr(audio_cfg, "mel_hop_size", None),
            mel_n_mels=int(getattr(audio_cfg, "mel_n_mels", 128)),
            rm_mode=str(getattr(audio_cfg, "rm_mode", "log_rms")),
            rm_weight=float(getattr(audio_cfg, "rm_weight", 1.0)),
            rm_loss_type=str(getattr(audio_cfg, "rm_loss_type", "L1")),
            rm_db_scale=bool(getattr(audio_cfg, "rm_db_scale", True)),
            eps=float(getattr(audio_cfg, "eps", 1e-7)),
        )

    if audio_loss_name == "piano_ssm_spectral":
        return _build_piano_ssm_spectral_loss_from_cfg(cfg, sample_rate=sample_rate)

    if audio_loss_name == "piano_ssm_spectral_plus_log_rms":
        audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_spectral_plus_log_rms")
        spectral_loss = _build_piano_ssm_spectral_loss_from_cfg(cfg, sample_rate=sample_rate)
        log_rms_loss = _build_log_rms_aux_loss_from_cfg(cfg)
        return PianoSSMSpectralPlusLogRMSLoss(
            spectral_loss=spectral_loss,
            log_rms_loss=log_rms_loss,
            spectral_weight=float(getattr(audio_cfg, "spectral_weight", 1.0)),
            log_rms_weight=float(getattr(audio_cfg, "log_rms_weight", 0.05)),
        )

    if audio_loss_name == "piano_ssm_spectral_plus_ddsp_loudness":
        audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_spectral_plus_ddsp_loudness")
        spectral_loss = _build_piano_ssm_spectral_loss_from_cfg(cfg, sample_rate=sample_rate)
        loudness_loss = _build_ddsp_piano_loudness_loss_from_cfg(
            cfg,
            sample_rate=sample_rate,
            frame_rate=int(frame_rate_override) if frame_rate_override is not None else None,
        )
        return PianoSSMSpectralPlusDDSPLoudnessLoss(
            spectral_loss=spectral_loss,
            loudness_loss=loudness_loss,
            spectral_weight=float(getattr(audio_cfg, "spectral_weight", 1.0)),
            loudness_weight=float(getattr(audio_cfg, "loudness_weight", 0.05)),
        )

    if audio_loss_name == "ddsp_piano_loudness":
        return _build_ddsp_piano_loudness_loss_from_cfg(
            cfg,
            sample_rate=sample_rate,
            frame_rate=int(frame_rate_override) if frame_rate_override is not None else None,
        )

    available = ", ".join(sorted(AUDIO_LOSS_TYPES.keys()))
    raise ValueError(f"Unknown proxy.audio_loss.type: {audio_loss_name!r}. Available: {available}")


# -----------------------------------------------------------------------------
# Loss function selector for velocity supervision
# -----------------------------------------------------------------------------


def get_loss_func(cfg, loss_type=None):
    """
    Return a callable with unified signature:
      fn(cfg, output_dict, target_dict, cond_dict=None) -> loss

    Selection order:
    - explicit loss_type if provided
    - cfg.loss.loss_type
    """
    selected_loss_type = loss_type if loss_type is not None else cfg.loss.loss_type
    loss_map = {
        "velocity_bce": velocity_bce,
        "velocity_mse": velocity_mse,
        "kim_bce_l1": kim_velocity_bce_l1,
    }
    if selected_loss_type in loss_map:
        return loss_map[selected_loss_type]

    raise ValueError(f"Incorrect loss_type: {selected_loss_type!r}")
