from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch


@dataclass(frozen=True)
class NoteLoudnessConfig:
    """Note-wise loudness proxy from the paper Route I / DiffProxy teacher target."""

    sample_rate: int = 22050
    n_fft: int = 2048
    hop: int = 256
    win_length: Optional[int] = None
    harmonic_count: int = 5
    band_bins: int = 1
    energy_window_s: float = 0.12
    onset_pre_s: float = 0.02
    onset_post_s: float = 0.08


@dataclass(frozen=True)
class NoteFeatureResult:
    times: np.ndarray
    freqs: np.ndarray
    magnitude: np.ndarray
    harmonic_energy: np.ndarray
    onset_flux: np.ndarray
    features: np.ndarray


class _NoteLikeProtocol:
    onset: float
    offset: float
    pitch: int


def load_audio_mono(audio_path: str, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(str(audio_path), sr=int(sample_rate), mono=True)
    if audio.ndim != 1:
        audio = np.mean(np.asarray(audio), axis=0)
    return np.asarray(audio, dtype=np.float32)


def _hz_from_midi(pitch: int) -> float:
    return 440.0 * (2.0 ** ((int(pitch) - 69) / 12.0))


def compute_magnitude_stft(
    audio: np.ndarray,
    *,
    sample_rate: int,
    cfg: NoteLoudnessConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if audio.ndim != 1:
        raise ValueError("audio must be 1-D mono waveform")
    win_length = int(cfg.win_length) if cfg.win_length else int(cfg.n_fft)
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    window = torch.hann_window(win_length)
    spec = torch.stft(
        waveform,
        n_fft=int(cfg.n_fft),
        hop_length=int(cfg.hop),
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs().cpu().numpy().astype(np.float64)  # (freq, time)
    times = np.arange(mag.shape[1], dtype=np.float64) * (float(cfg.hop) / float(sample_rate))
    freqs = np.linspace(0.0, sample_rate / 2.0, mag.shape[0], dtype=np.float64)
    return times, freqs, mag


def _note_window_indices(times: np.ndarray, t0: float, t1: float) -> Tuple[int, int]:
    i0 = int(np.searchsorted(times, max(float(t0), 0.0), side="left"))
    i1 = int(np.searchsorted(times, float(t1), side="right"))
    i0 = max(0, min(i0, len(times)))
    i1 = max(0, min(i1, len(times)))
    return i0, i1


def extract_note_loudness_features(
    audio: np.ndarray,
    notes: Sequence[_NoteLikeProtocol],
    *,
    sample_rate: int,
    cfg: Optional[NoteLoudnessConfig] = None,
) -> NoteFeatureResult:
    cfg = cfg or NoteLoudnessConfig(sample_rate=int(sample_rate))
    times, freqs, mag = compute_magnitude_stft(audio, sample_rate=sample_rate, cfg=cfg)
    mag_sq = mag ** 2

    if mag.shape[1] >= 2:
        diff = mag[:, 1:] - mag[:, :-1]
        flux_per_frame = np.maximum(diff, 0.0).sum(axis=0)
        flux_per_frame = np.concatenate([np.zeros(1, dtype=np.float64), flux_per_frame], axis=0)
    else:
        flux_per_frame = np.zeros(mag.shape[1], dtype=np.float64)

    freq_res = float(sample_rate) / float(cfg.n_fft)
    harmonic_energy = np.full(len(notes), np.nan, dtype=np.float64)
    onset_flux = np.full(len(notes), np.nan, dtype=np.float64)

    for idx, note in enumerate(notes):
        onset_s = float(note.onset)
        if not np.isfinite(onset_s):
            continue

        # Harmonic energy shortly after onset.
        e0 = onset_s
        e1 = onset_s + float(cfg.energy_window_s)
        i0, i1 = _note_window_indices(times, e0, e1)
        if i1 <= i0:
            continue

        f0 = _hz_from_midi(int(note.pitch))
        energy = 0.0
        for h in range(1, int(cfg.harmonic_count) + 1):
            fh = f0 * float(h)
            if fh >= freqs[-1]:
                break
            center_bin = int(round(fh / max(freq_res, 1e-12)))
            if center_bin < 0 or center_bin >= mag.shape[0]:
                continue
            lo = max(0, center_bin - int(cfg.band_bins))
            hi = min(mag.shape[0], center_bin + int(cfg.band_bins) + 1)
            energy += float(np.mean(mag_sq[lo:hi, i0:i1]))
        harmonic_energy[idx] = np.log1p(max(energy, 0.0))

        # Onset spectral flux around onset.
        o0 = onset_s - float(cfg.onset_pre_s)
        o1 = onset_s + float(cfg.onset_post_s)
        j0, j1 = _note_window_indices(times, o0, o1)
        if j1 > j0:
            onset_flux[idx] = np.log1p(float(np.mean(flux_per_frame[j0:j1])))

    features = np.stack([harmonic_energy, onset_flux], axis=1)
    return NoteFeatureResult(
        times=times,
        freqs=freqs,
        magnitude=mag,
        harmonic_energy=harmonic_energy,
        onset_flux=onset_flux,
        features=features,
    )


def extract_note_loudness_from_files(
    audio_path: str,
    notes: Sequence[_NoteLikeProtocol],
    *,
    sample_rate: int,
    cfg: Optional[NoteLoudnessConfig] = None,
) -> NoteFeatureResult:
    audio = load_audio_mono(str(audio_path), int(sample_rate))
    return extract_note_loudness_features(audio, notes, sample_rate=sample_rate, cfg=cfg)
