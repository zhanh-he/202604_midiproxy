from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

from .common import load_json


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class VelocityDistributionMapping:
    """Empirical inverse-CDF mapping from percentile to MIDI velocity.

    The JSON format matches data_analysis/stats/midi_sampler/*.json.
    """

    midi_values: np.ndarray
    counts: np.ndarray
    cdf: np.ndarray
    dataset_name: str
    observed_min: int
    observed_max: int
    boundaries_01: Tuple[float, float]
    source_json: Optional[Path] = None

    @classmethod
    def from_stats_json(cls, stats_json: str | Path) -> "VelocityDistributionMapping":
        payload = load_json(stats_json)
        if "velocities_01" not in payload:
            raise KeyError(f"'velocities_01' not found in stats JSON: {stats_json}")

        velocity_block = payload["velocities_01"]
        values_01 = np.asarray(velocity_block.get("values", []), dtype=np.float64)
        counts = np.asarray(velocity_block.get("counts", []), dtype=np.float64)
        if values_01.size == 0 or counts.size == 0:
            raise ValueError(f"Velocity histogram is empty in: {stats_json}")
        if values_01.shape != counts.shape:
            raise ValueError("Velocity histogram values/counts shape mismatch.")

        midi_values = np.clip(np.rint(values_01 * 127.0), 1.0, 127.0).astype(np.int64)
        counts = np.asarray(counts, dtype=np.float64)
        total = float(np.sum(counts))
        if total <= 0:
            raise ValueError("Velocity histogram counts sum to zero.")
        cdf = np.cumsum(counts) / total

        boundaries = payload.get("velocity_boundaries_01", [0.33, 0.66])
        if len(boundaries) != 2:
            boundaries = [0.33, 0.66]

        observed_range = payload.get("velocity_range_observed") or [int(midi_values.min()), int(midi_values.max())]
        if len(observed_range) != 2:
            observed_range = [int(midi_values.min()), int(midi_values.max())]

        return cls(
            midi_values=midi_values,
            counts=counts,
            cdf=cdf,
            dataset_name=str(payload.get("dataset", Path(stats_json).stem)),
            observed_min=int(observed_range[0]),
            observed_max=int(observed_range[1]),
            boundaries_01=(float(boundaries[0]), float(boundaries[1])),
            source_json=Path(stats_json),
        )

    @classmethod
    def from_velocity_array(
        cls,
        velocities: ArrayLike,
        *,
        dataset_name: str = "custom",
    ) -> "VelocityDistributionMapping":
        arr = np.asarray(velocities, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("No finite velocities provided.")
        arr = np.clip(np.rint(arr), 1.0, 127.0).astype(np.int64)
        unique, counts = np.unique(arr, return_counts=True)
        cdf = np.cumsum(counts.astype(np.float64)) / float(np.sum(counts))
        return cls(
            midi_values=unique,
            counts=counts.astype(np.float64),
            cdf=cdf,
            dataset_name=dataset_name,
            observed_min=int(unique.min()),
            observed_max=int(unique.max()),
            boundaries_01=(
                float(np.quantile(arr / 127.0, 0.33)),
                float(np.quantile(arr / 127.0, 0.66)),
            ),
            source_json=None,
        )

    def map_percentiles(self, percentiles: ArrayLike) -> np.ndarray:
        p = np.asarray(percentiles, dtype=np.float64)
        clipped = np.clip(p, 0.0, 1.0)
        idx = np.searchsorted(self.cdf, clipped, side="left")
        idx = np.clip(idx, 0, len(self.midi_values) - 1)
        return self.midi_values[idx].astype(np.int64)

    def summary(self) -> Dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "observed_min": int(self.observed_min),
            "observed_max": int(self.observed_max),
            "boundaries_01": [float(self.boundaries_01[0]), float(self.boundaries_01[1])],
            "num_support_values": int(self.midi_values.size),
            "source_json": str(self.source_json) if self.source_json else None,
        }


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j)
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def percentile_rank(values: ArrayLike) -> np.ndarray:
    """Average-tie percentile rank in [0, 1] with NaN passthrough."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    valid = arr[mask]
    if valid.size == 1:
        out[mask] = 0.5
        return out
    ranks = _rankdata_average(valid)
    out[mask] = ranks / float(valid.size - 1)
    return out


def robust_unit_scale(
    values: ArrayLike,
    *,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    valid = arr[mask]
    lo = float(np.quantile(valid, q_low))
    hi = float(np.quantile(valid, q_high))
    if hi <= lo + 1e-12:
        out[mask] = 0.5
        return out
    out[mask] = np.clip((valid - lo) / (hi - lo), 0.0, 1.0)
    return out


def combine_note_loudness_to_percentiles(
    harmonic_energy: ArrayLike,
    onset_flux: ArrayLike,
    *,
    mode: Literal["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"] = "rank_blend",
    harmonic_weight: float = 1.0,
    flux_weight: float = 0.25,
) -> np.ndarray:
    """Combine per-note loudness parameters into one percentile-like score.

    This is intentionally training-free. The default `rank_blend` makes the
    mapping robust to scale mismatch across recordings and instruments.
    """
    e = np.asarray(harmonic_energy, dtype=np.float64).reshape(-1)
    x = np.asarray(onset_flux, dtype=np.float64).reshape(-1)
    if e.shape != x.shape:
        raise ValueError("harmonic_energy and onset_flux must have the same shape.")

    mode = str(mode).strip().lower()
    if harmonic_weight < 0 or flux_weight < 0:
        raise ValueError("Weights must be non-negative.")

    if mode == "rank_harmonic":
        return percentile_rank(e)
    if mode == "rank_flux":
        return percentile_rank(x)

    if mode == "robust_blend":
        e_score = robust_unit_scale(e)
        x_score = robust_unit_scale(x)
    elif mode == "rank_blend":
        e_score = percentile_rank(e)
        x_score = percentile_rank(x)
    else:
        raise ValueError(f"Unsupported combination mode: {mode}")

    w_sum = float(harmonic_weight + flux_weight)
    if w_sum <= 1e-12:
        raise ValueError("At least one weight must be positive.")

    out = np.full(e.shape, np.nan, dtype=np.float64)
    mask_e = np.isfinite(e_score)
    mask_x = np.isfinite(x_score)
    both = mask_e & mask_x
    only_e = mask_e & ~mask_x
    only_x = mask_x & ~mask_e

    if np.any(both):
        out[both] = (harmonic_weight * e_score[both] + flux_weight * x_score[both]) / w_sum
    if np.any(only_e):
        out[only_e] = e_score[only_e]
    if np.any(only_x):
        out[only_x] = x_score[only_x]
    return out


def map_note_loudness_to_midi_velocity(
    harmonic_energy: ArrayLike,
    onset_flux: ArrayLike,
    mapping: VelocityDistributionMapping,
    *,
    mode: Literal["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"] = "rank_blend",
    harmonic_weight: float = 1.0,
    flux_weight: float = 0.25,
    fallback_velocity: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    percentiles = combine_note_loudness_to_percentiles(
        harmonic_energy,
        onset_flux,
        mode=mode,
        harmonic_weight=harmonic_weight,
        flux_weight=flux_weight,
    )
    velocities = mapping.map_percentiles(np.nan_to_num(percentiles, nan=0.5))
    if fallback_velocity is not None:
        invalid = ~np.isfinite(percentiles)
        velocities[invalid] = int(np.clip(round(fallback_velocity), 1, 127))
    return velocities.astype(np.int64), percentiles
