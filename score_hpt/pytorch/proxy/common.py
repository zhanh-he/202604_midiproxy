from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))



def resolve_backend_segment_seconds(
    cfg,
    backend_cfg=None,
    backend_default: float = 0.0,
    total_segment_seconds: Optional[float] = None,
) -> float:
    """Resolve the backend segment length used inside a fixed Score-HPT segment.

    Priority:
      1. proxy.backend_segment_seconds
      2. legacy proxy.crop_seconds
      3. backend-specific native/segment defaults
      4. explicit backend_default

    The returned value is capped to total_segment_seconds when provided.
    """
    resolved = 0.0
    proxy_cfg = getattr(cfg, 'proxy', None)
    for key in ('backend_segment_seconds', 'crop_seconds'):
        raw = getattr(proxy_cfg, key, 0.0) if proxy_cfg is not None else 0.0
        try:
            value = float(raw or 0.0)
        except Exception:
            value = 0.0
        if value > 0:
            resolved = value
            break

    if resolved <= 0 and backend_cfg is not None:
        for key in ('backend_segment_seconds', 'native_segment_seconds', 'segment_seconds'):
            raw = getattr(backend_cfg, key, 0.0)
            try:
                value = float(raw or 0.0)
            except Exception:
                value = 0.0
            if value > 0:
                resolved = value
                break

    if resolved <= 0:
        resolved = float(backend_default or 0.0)

    if total_segment_seconds is not None and float(total_segment_seconds) > 0:
        resolved = min(float(resolved), float(total_segment_seconds))
    return float(resolved)



def resolve_supervision_sample_rate(cfg) -> int:
    supervision_cfg = getattr(getattr(cfg, 'proxy', None), 'supervision', None)
    value = int(getattr(supervision_cfg, 'sample_rate', 0) or 0)
    if value > 0:
        return value
    return int(getattr(getattr(cfg, 'feature', None), 'sample_rate', 16000) or 16000)



def resolve_supervision_frame_rate(cfg) -> float:
    supervision_cfg = getattr(getattr(cfg, 'proxy', None), 'supervision', None)
    value = float(getattr(supervision_cfg, 'frame_rate', 0.0) or 0.0)
    if value > 0:
        return value
    return float(getattr(getattr(cfg, 'feature', None), 'frames_per_second', 100.0) or 100.0)



def resolve_supervision_fft_size(cfg) -> int:
    supervision_cfg = getattr(getattr(cfg, 'proxy', None), 'supervision', None)
    value = int(getattr(supervision_cfg, 'fft_size', 0) or 0)
    if value > 0:
        return value
    return int(getattr(getattr(cfg, 'feature', None), 'fft_size', 2048) or 2048)



def resolve_supervision_hop_size(cfg, sample_rate: Optional[int] = None, frame_rate: Optional[float] = None) -> int:
    supervision_cfg = getattr(getattr(cfg, 'proxy', None), 'supervision', None)
    value = int(getattr(supervision_cfg, 'hop_size', 0) or 0)
    if value > 0:
        return value
    sr = int(sample_rate or resolve_supervision_sample_rate(cfg))
    fps = float(frame_rate or resolve_supervision_frame_rate(cfg))
    return max(1, _round_half_up(sr / max(fps, 1e-6)))


def get_velocity_prediction(output_dict: dict) -> torch.Tensor:
    """Return the main normalized velocity prediction in [0, 1]."""
    if "vel_corr" in output_dict:
        return output_dict["vel_corr"]
    if "velocity_output" in output_dict:
        return output_dict["velocity_output"]
    raise KeyError("velocity prediction not found in output_dict")



def choose_crop_bounds(
    total_seconds: float,
    crop_seconds: float,
    mode: str = "random",
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[float, float]:
    """Choose a crop window in seconds."""
    total_seconds = float(total_seconds)
    crop_seconds = float(crop_seconds)
    mode = str(mode or "random").strip().lower()
    if crop_seconds <= 0 or crop_seconds >= total_seconds:
        return 0.0, total_seconds

    max_start = max(0.0, total_seconds - crop_seconds)
    if mode in ("full", "none"):
        return 0.0, total_seconds
    if mode == "center":
        return 0.5 * max_start, crop_seconds
    if mode == "random":
        rng = random_state if random_state is not None else np.random
        return float(rng.uniform(0.0, max_start)), crop_seconds
    raise ValueError(f"Unknown crop mode: {mode}")



def crop_audio(audio: torch.Tensor, sample_rate: int, start_sec: float, crop_sec: float) -> torch.Tensor:
    """Crop waveform [B, samples] with second-based boundaries."""
    if crop_sec <= 0:
        return audio
    start_sample = int(round(start_sec * sample_rate))
    crop_samples = int(round(crop_sec * sample_rate))
    end_sample = min(audio.size(-1), start_sample + crop_samples)
    if end_sample - start_sample < crop_samples:
        start_sample = max(0, end_sample - crop_samples)
    return audio[..., start_sample:end_sample]



def crop_roll_time_first(
    roll: torch.Tensor,
    frames_per_second: float,
    start_sec: float,
    crop_sec: float,
    include_endpoint: bool = True,
) -> torch.Tensor:
    """Crop tensors with time axis at dim=1, e.g. [B, T, P]."""
    if crop_sec <= 0:
        return roll
    start_frame = int(round(start_sec * frames_per_second))
    crop_frames = int(round(crop_sec * frames_per_second)) + (1 if include_endpoint else 0)
    end_frame = min(roll.size(1), start_frame + crop_frames)
    if end_frame - start_frame < crop_frames:
        start_frame = max(0, end_frame - crop_frames)
    return roll[:, start_frame:end_frame, ...]



def crop_roll_time_last(
    roll: torch.Tensor,
    frames_per_second: float,
    start_sec: float,
    crop_sec: float,
    include_endpoint: bool = False,
) -> torch.Tensor:
    """Crop tensors with time axis at dim=-1, e.g. [B, C, T]."""
    if crop_sec <= 0:
        return roll
    start_frame = int(round(start_sec * frames_per_second))
    crop_frames = int(round(crop_sec * frames_per_second)) + (1 if include_endpoint else 0)
    end_frame = min(roll.size(-1), start_frame + crop_frames)
    if end_frame - start_frame < crop_frames:
        start_frame = max(0, end_frame - crop_frames)
    return roll[..., start_frame:end_frame]



def resample_waveform(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """Linear resampling based on torch.interpolate for mono waveforms [B, S]."""
    if src_sr == dst_sr:
        return waveform
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    target_len = max(1, int(round(waveform.size(-1) * float(dst_sr) / float(src_sr))))
    y = F.interpolate(
        waveform.unsqueeze(1),
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return y.squeeze(1)



def resample_roll_btp(roll: torch.Tensor, target_frames: int, mode: str = "nearest") -> torch.Tensor:
    """Resample [B, T, P] (or [B, T, 1]) to a new time resolution."""
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if roll.size(1) == target_frames:
        return roll

    x = roll.transpose(1, 2)
    if mode == "nearest":
        y = F.interpolate(x, size=target_frames, mode=mode)
    else:
        y = F.interpolate(x, size=target_frames, mode=mode, align_corners=False)
    return y.transpose(1, 2)


def resample_roll_bct(roll: torch.Tensor, target_frames: int, mode: str = "nearest") -> torch.Tensor:
    """Resample [B, C, T] tensors to a new time resolution along the last axis."""
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if roll.size(-1) == target_frames:
        return roll

    if mode == "nearest":
        return F.interpolate(roll, size=target_frames, mode=mode)
    return F.interpolate(roll, size=target_frames, mode=mode, align_corners=False)



def align_time_dim(*tensors: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
    """Trim all tensors with time axis at dim=1 to the shared minimum."""
    valid = [tensor for tensor in tensors if tensor is not None and torch.is_tensor(tensor)]
    if not valid:
        return tensors
    min_steps = min(tensor.size(1) for tensor in valid)
    out = []
    for tensor in tensors:
        if tensor is None or not torch.is_tensor(tensor):
            out.append(tensor)
        else:
            out.append(tensor[:, :min_steps])
    return tuple(out)



def stable_voice_assignment(active_pitch_roll: np.ndarray, n_voices: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign active pitches to a fixed number of channels while preserving channel continuity.

    Args:
        active_pitch_roll: [T, 88] MIDI pitch values on active frames, else 0.
        n_voices: number of proxy voices.

    Returns:
        assigned_pitch: [T, n_voices] MIDI pitch value per channel.
        assigned_index: [T, n_voices] original pitch-bin index per channel, -1 for silence.
    """
    if active_pitch_roll.ndim != 2:
        raise ValueError(f"Expected active_pitch_roll with 2 dims, got {active_pitch_roll.shape}")

    steps, _ = active_pitch_roll.shape
    assigned_pitch = np.zeros((steps, n_voices), dtype=np.float32)
    assigned_index = np.full((steps, n_voices), -1, dtype=np.int64)

    prev_pitch = np.zeros(n_voices, dtype=np.int32)
    next_assigner = 0

    for t in range(steps):
        row = np.asarray(active_pitch_roll[t], dtype=np.float32)
        active_idx = np.flatnonzero(row > 0)
        if active_idx.size > 0:
            pitch_vals = row[active_idx].astype(np.int32)
            order = np.argsort(pitch_vals)
            active_idx = active_idx[order]
            pitch_vals = pitch_vals[order]
            if active_idx.size > n_voices:
                active_idx = active_idx[-n_voices:]
                pitch_vals = pitch_vals[-n_voices:]
        else:
            pitch_vals = np.array([], dtype=np.int32)

        pitch_to_idx = {int(p): int(i) for p, i in zip(pitch_vals.tolist(), active_idx.tolist())}

        cur_pitch = np.zeros(n_voices, dtype=np.int32)
        cur_index = np.full(n_voices, -1, dtype=np.int64)

        used = set()
        # Keep sustained notes on the same channel.
        for ch in range(n_voices):
            pitch = int(prev_pitch[ch])
            if pitch > 0 and pitch in pitch_to_idx and pitch not in used:
                cur_pitch[ch] = pitch
                cur_index[ch] = pitch_to_idx[pitch]
                used.add(pitch)

        remaining = [(int(p), pitch_to_idx[int(p)]) for p in pitch_vals.tolist() if int(p) not in used]
        free_channels = [ch for ch in range(n_voices) if cur_pitch[ch] == 0]
        if free_channels:
            if next_assigner in free_channels:
                pivot = free_channels.index(next_assigner)
                free_channels = free_channels[pivot:] + free_channels[:pivot]
            for (pitch, idx), ch in zip(remaining, free_channels):
                cur_pitch[ch] = pitch
                cur_index[ch] = idx
                next_assigner = (ch + 1) % n_voices

        assigned_pitch[t] = cur_pitch.astype(np.float32)
        assigned_index[t] = cur_index
        prev_pitch = cur_pitch

    return assigned_pitch, assigned_index



def derive_channel_onsets(assigned_pitch: torch.Tensor) -> torch.Tensor:
    """Return [T, C] onset mask from channel pitch changes."""
    if assigned_pitch.dim() != 2:
        raise ValueError(f"Expected assigned_pitch with 2 dims, got {assigned_pitch.shape}")
    onsets = torch.zeros_like(assigned_pitch)
    if assigned_pitch.numel() == 0:
        return onsets
    onsets[0] = (assigned_pitch[0] > 0).to(assigned_pitch.dtype)
    if assigned_pitch.size(0) > 1:
        changed = assigned_pitch[1:] != assigned_pitch[:-1]
        onsets[1:] = ((assigned_pitch[1:] > 0) & changed).to(assigned_pitch.dtype)
    return onsets
