from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from direct_invension.common import dump_json, ensure_repo_imports, slugify
    from direct_invension.midi_tools import extract_sorted_notes, replace_note_velocities, write_flat_velocity_copy
    from direct_invension.midi_velocity_mapping import VelocityDistributionMapping, map_note_loudness_to_midi_velocity
    from direct_invension.note_loudness import NoteLoudnessConfig, extract_note_loudness_from_files
else:
    from .common import dump_json, ensure_repo_imports, slugify
    from .midi_tools import extract_sorted_notes, replace_note_velocities, write_flat_velocity_copy
    from .midi_velocity_mapping import VelocityDistributionMapping, map_note_loudness_to_midi_velocity
    from .note_loudness import NoteLoudnessConfig, extract_note_loudness_from_files

from dataclasses import asdict, dataclass
import argparse
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Route1Config:
    dataset_stats_json: Optional[str] = None
    mapping_mode: Literal["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"] = "rank_blend"
    harmonic_weight: float = 1.0
    flux_weight: float = 0.25
    fallback_velocity: int = 64
    flat_velocity: int = 64
    note_sample_rate: int = 22050
    note_fft_size: int = 2048
    note_hop: int = 256
    harmonic_count: int = 5
    band_bins: int = 1
    energy_window_s: float = 0.12
    onset_pre_s: float = 0.02
    onset_post_s: float = 0.08

    def note_feature_config(self) -> NoteLoudnessConfig:
        return NoteLoudnessConfig(
            sample_rate=int(self.note_sample_rate),
            n_fft=int(self.note_fft_size),
            hop=int(self.note_hop),
            harmonic_count=int(self.harmonic_count),
            band_bins=int(self.band_bins),
            energy_window_s=float(self.energy_window_s),
            onset_pre_s=float(self.onset_pre_s),
            onset_post_s=float(self.onset_post_s),
        )


def _relative_key(dataset_dir: Path, midi_path: Path) -> str:
    try:
        rel = midi_path.relative_to(dataset_dir)
        return str(rel.with_suffix(""))
    except Exception:
        return midi_path.stem


def _route1_out_paths(
    out_dir: Path,
    dataset_dir: Optional[Path],
    midi_path: Path,
    *,
    flat_velocity: int,
) -> Tuple[Path, Path, Path]:
    if dataset_dir is not None:
        try:
            rel = midi_path.relative_to(dataset_dir)
        except Exception:
            rel = Path(midi_path.name)
    else:
        rel = Path(midi_path.name)
    pred_midi = (out_dir / "pred_midis" / rel).with_suffix(".direct.mid")
    flat_midi = (out_dir / "flat_midis" / rel).with_suffix(f".flat{int(flat_velocity)}.mid")
    feature_json = (out_dir / "note_features" / rel).with_suffix(".route1.json")
    return pred_midi, flat_midi, feature_json


def _feature_summary(harmonic_energy: np.ndarray, onset_flux: np.ndarray, percentiles: np.ndarray, velocities: np.ndarray) -> Dict[str, object]:
    def stats(values: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "num_notes": int(len(velocities)),
        "harmonic_energy": stats(harmonic_energy),
        "onset_flux": stats(onset_flux),
        "combined_percentile": stats(percentiles),
        "pred_velocity": {
            "mean": float(np.mean(velocities)) if len(velocities) else float("nan"),
            "std": float(np.std(velocities)) if len(velocities) else float("nan"),
            "min": int(np.min(velocities)) if len(velocities) else 0,
            "max": int(np.max(velocities)) if len(velocities) else 0,
        },
    }


def predict_direct_inversion_for_pair(
    *,
    audio_path: str | Path,
    midi_path: str | Path,
    output_midi_path: str | Path,
    config: Route1Config,
    dataset_mapping: Optional[VelocityDistributionMapping] = None,
    feature_json_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Route I: note-wise loudness extraction -> direct MIDI velocity inversion."""
    midi_path = Path(midi_path)
    audio_path = Path(audio_path)
    output_midi_path = Path(output_midi_path)

    if dataset_mapping is None:
        if not config.dataset_stats_json:
            raise ValueError("Route1Config.dataset_stats_json is required when dataset_mapping is not provided.")
        dataset_mapping = VelocityDistributionMapping.from_stats_json(config.dataset_stats_json)

    notes = extract_sorted_notes(midi_path)
    note_features = extract_note_loudness_from_files(
        str(audio_path),
        notes,
        sample_rate=int(config.note_sample_rate),
        cfg=config.note_feature_config(),
    )

    velocities, percentiles = map_note_loudness_to_midi_velocity(
        note_features.harmonic_energy,
        note_features.onset_flux,
        dataset_mapping,
        mode=config.mapping_mode,
        harmonic_weight=float(config.harmonic_weight),
        flux_weight=float(config.flux_weight),
        fallback_velocity=int(config.fallback_velocity),
    )
    replace_note_velocities(midi_path, velocities, output_midi_path)

    summary = _feature_summary(
        note_features.harmonic_energy,
        note_features.onset_flux,
        percentiles,
        velocities,
    )
    payload = {
        "route": "route1_direct_inversion",
        "audio_path": str(audio_path),
        "midi_path": str(midi_path),
        "output_midi_path": str(output_midi_path),
        "mapping": dataset_mapping.summary(),
        "config": asdict(config),
        "summary": summary,
    }

    if feature_json_path is not None:
        feature_json_path = Path(feature_json_path)
        payload["feature_json_path"] = str(dump_json(feature_json_path, payload))
    return payload


def _build_dataset_scan_helpers():
    ensure_repo_imports()
    from data_analysis.cli._dataset_utils import (  # type: ignore
        load_maestro_audio_map,
        resolve_real_audio,
        scan_midis,
    )
    return scan_midis, load_maestro_audio_map, resolve_real_audio


def predict_route1_dataset(
    *,
    dataset_type: str,
    dataset_dir: str | Path,
    out_dir: str | Path,
    dataset_stats_json: str | Path,
    split: str = "test",
    maps_pianos: str = "both",
    flat_velocity: int = 64,
    mapping_mode: Literal["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"] = "rank_blend",
    harmonic_weight: float = 1.0,
    flux_weight: float = 0.25,
    fallback_velocity: int = 64,
    note_sample_rate: int = 22050,
    note_fft_size: int = 2048,
    note_hop: int = 256,
    harmonic_count: int = 5,
    band_bins: int = 1,
    energy_window_s: float = 0.12,
    onset_pre_s: float = 0.02,
    onset_post_s: float = 0.08,
    skip_existing: bool = True,
) -> Dict[str, object]:
    """Run Route I over a dataset and create manifests for direct + flat baselines."""
    scan_midis, load_maestro_audio_map, resolve_real_audio = _build_dataset_scan_helpers()

    dataset_dir = Path(dataset_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = VelocityDistributionMapping.from_stats_json(dataset_stats_json)
    cfg = Route1Config(
        dataset_stats_json=str(dataset_stats_json),
        mapping_mode=mapping_mode,
        harmonic_weight=float(harmonic_weight),
        flux_weight=float(flux_weight),
        fallback_velocity=int(fallback_velocity),
        flat_velocity=int(flat_velocity),
        note_sample_rate=int(note_sample_rate),
        note_fft_size=int(note_fft_size),
        note_hop=int(note_hop),
        harmonic_count=int(harmonic_count),
        band_bins=int(band_bins),
        energy_window_s=float(energy_window_s),
        onset_pre_s=float(onset_pre_s),
        onset_post_s=float(onset_post_s),
    )

    midi_files = scan_midis(
        dataset_type,
        dataset_dir,
        split=split,
        maps_pianos=maps_pianos,
    )
    maestro_audio_map = load_maestro_audio_map(dataset_type, dataset_dir, split=split)

    direct_items: List[Dict[str, object]] = []
    flat_items: List[Dict[str, object]] = []
    file_summaries: List[Dict[str, object]] = []

    for midi_path in midi_files:
        midi_path = Path(midi_path).resolve()
        real_audio = resolve_real_audio(dataset_type, dataset_dir, midi_path, maestro_audio_map)
        key = _relative_key(dataset_dir, midi_path)
        pred_midi_path, flat_midi_path, feature_json_path = _route1_out_paths(
            out_dir,
            dataset_dir,
            midi_path,
            flat_velocity=int(flat_velocity),
        )

        pair_payload: Dict[str, object]
        if skip_existing and pred_midi_path.exists():
            pair_payload = {
                "route": "route1_direct_inversion",
                "audio_path": str(real_audio),
                "midi_path": str(midi_path),
                "output_midi_path": str(pred_midi_path),
                "mapping": mapping.summary(),
                "config": asdict(cfg),
                "summary": {"num_notes": len(extract_sorted_notes(midi_path)), "skipped_existing": True},
                "feature_json_path": str(feature_json_path) if feature_json_path.exists() else None,
            }
        else:
            pair_payload = predict_direct_inversion_for_pair(
                audio_path=real_audio,
                midi_path=midi_path,
                output_midi_path=pred_midi_path,
                config=cfg,
                dataset_mapping=mapping,
                feature_json_path=feature_json_path,
            )

        if not (skip_existing and flat_midi_path.exists()):
            write_flat_velocity_copy(midi_path, flat_midi_path, flat_velocity=int(flat_velocity))

        direct_items.append(
            {
                "key": key,
                "label": "route1_direct_inversion",
                "gt_midi": str(midi_path),
                "pred_midi": str(pred_midi_path),
                "real_audio": str(real_audio),
                "feature_json": str(feature_json_path) if feature_json_path.exists() else pair_payload.get("feature_json_path"),
            }
        )
        flat_items.append(
            {
                "key": key,
                "label": f"flat_velocity_{int(flat_velocity)}",
                "gt_midi": str(midi_path),
                "pred_midi": str(flat_midi_path),
                "real_audio": str(real_audio),
            }
        )
        file_summaries.append(pair_payload)

    direct_manifest = {
        "format_version": 1,
        "label": "route1_direct_inversion",
        "dataset_type": str(dataset_type),
        "dataset_dir": str(dataset_dir),
        "split": split,
        "maps_pianos": maps_pianos,
        "config": asdict(cfg),
        "items": direct_items,
    }
    flat_manifest = {
        "format_version": 1,
        "label": f"flat_velocity_{int(flat_velocity)}",
        "dataset_type": str(dataset_type),
        "dataset_dir": str(dataset_dir),
        "split": split,
        "maps_pianos": maps_pianos,
        "config": asdict(cfg),
        "items": flat_items,
    }
    direct_manifest_path = dump_json(out_dir / "route1_direct_manifest.json", direct_manifest)
    flat_manifest_path = dump_json(out_dir / f"route1_flat{int(flat_velocity)}_manifest.json", flat_manifest)
    file_summary_path = dump_json(out_dir / "route1_direct_predictions.json", {"files": file_summaries})

    return {
        "dataset_type": dataset_type,
        "dataset_dir": str(dataset_dir),
        "num_items": len(direct_items),
        "out_dir": str(out_dir),
        "direct_manifest_path": str(direct_manifest_path),
        "flat_manifest_path": str(flat_manifest_path),
        "file_summary_path": str(file_summary_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route I direct inversion utilities.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_pair = subparsers.add_parser("pair", help="Run Route I on one audio+MIDI pair.")
    p_pair.add_argument("--audio", required=True)
    p_pair.add_argument("--midi", required=True)
    p_pair.add_argument("--stats_json", required=True)
    p_pair.add_argument("--out_midi", required=True)
    p_pair.add_argument("--feature_json", default=None)
    p_pair.add_argument("--mapping_mode", default="rank_blend", choices=["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"])
    p_pair.add_argument("--harmonic_weight", type=float, default=1.0)
    p_pair.add_argument("--flux_weight", type=float, default=0.25)
    p_pair.add_argument("--fallback_velocity", type=int, default=64)
    p_pair.add_argument("--note_sr", type=int, default=22050)
    p_pair.add_argument("--note_fft", type=int, default=2048)
    p_pair.add_argument("--note_hop", type=int, default=256)
    p_pair.add_argument("--harmonics", type=int, default=5)
    p_pair.add_argument("--band_bins", type=int, default=1)

    p_dataset = subparsers.add_parser("dataset", help="Run Route I on a dataset and create manifests.")
    p_dataset.add_argument("--dataset_type", required=True, choices=["smd", "maestro", "maps"])
    p_dataset.add_argument("--dataset_dir", required=True)
    p_dataset.add_argument("--stats_json", required=True)
    p_dataset.add_argument("--out_dir", required=True)
    p_dataset.add_argument("--split", default="test")
    p_dataset.add_argument("--maps_pianos", default="both")
    p_dataset.add_argument("--flat_velocity", type=int, default=64)
    p_dataset.add_argument("--mapping_mode", default="rank_blend", choices=["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"])
    p_dataset.add_argument("--harmonic_weight", type=float, default=1.0)
    p_dataset.add_argument("--flux_weight", type=float, default=0.25)
    p_dataset.add_argument("--fallback_velocity", type=int, default=64)
    p_dataset.add_argument("--note_sr", type=int, default=22050)
    p_dataset.add_argument("--note_fft", type=int, default=2048)
    p_dataset.add_argument("--note_hop", type=int, default=256)
    p_dataset.add_argument("--harmonics", type=int, default=5)
    p_dataset.add_argument("--band_bins", type=int, default=1)
    p_dataset.add_argument("--energy_window_s", type=float, default=0.12)
    p_dataset.add_argument("--onset_pre_s", type=float, default=0.02)
    p_dataset.add_argument("--onset_post_s", type=float, default=0.08)
    p_dataset.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.cmd == "pair":
        cfg = Route1Config(
            dataset_stats_json=args.stats_json,
            mapping_mode=args.mapping_mode,
            harmonic_weight=args.harmonic_weight,
            flux_weight=args.flux_weight,
            fallback_velocity=args.fallback_velocity,
            note_sample_rate=args.note_sr,
            note_fft_size=args.note_fft,
            note_hop=args.note_hop,
            harmonic_count=args.harmonics,
            band_bins=args.band_bins,
        )
        payload = predict_direct_inversion_for_pair(
            audio_path=args.audio,
            midi_path=args.midi,
            output_midi_path=args.out_midi,
            config=cfg,
            feature_json_path=args.feature_json,
        )
        print(payload)
        return

    payload = predict_route1_dataset(
        dataset_type=args.dataset_type,
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        dataset_stats_json=args.stats_json,
        split=args.split,
        maps_pianos=args.maps_pianos,
        flat_velocity=args.flat_velocity,
        mapping_mode=args.mapping_mode,
        harmonic_weight=args.harmonic_weight,
        flux_weight=args.flux_weight,
        fallback_velocity=args.fallback_velocity,
        note_sample_rate=args.note_sr,
        note_fft_size=args.note_fft,
        note_hop=args.note_hop,
        harmonic_count=args.harmonics,
        band_bins=args.band_bins,
        energy_window_s=args.energy_window_s,
        onset_pre_s=args.onset_pre_s,
        onset_post_s=args.onset_post_s,
        skip_existing=not args.overwrite,
    )
    print(payload)


if __name__ == "__main__":
    main()
