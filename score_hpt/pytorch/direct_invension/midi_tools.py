from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SortedMidiNote:
    onset: float
    offset: float
    pitch: int
    velocity: int
    instrument_index: int
    note_index: int


@dataclass(frozen=True)
class VelocityAlignmentResult:
    num_gt_notes: int
    num_pred_notes: int
    num_matched_notes: int
    mae: float
    matched_in_exact_order: bool
    unmatched_gt: int
    unmatched_pred: int


def _import_pretty_midi():
    try:
        import pretty_midi  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "pretty_midi is required for Route I direct inversion and evaluation. "
            "Install it with `pip install pretty_midi`."
        ) from exc
    return pretty_midi


def load_pretty_midi(midi_path: str | Path):
    pretty_midi = _import_pretty_midi()
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")
    return pretty_midi.PrettyMIDI(str(midi_path))


def _sorted_note_refs(pm) -> List[Tuple[Tuple[float, int, float, int, int], object]]:
    refs: List[Tuple[Tuple[float, int, float, int, int], object]] = []
    for inst_idx, inst in enumerate(pm.instruments):
        for note_idx, note in enumerate(inst.notes):
            key = (
                round(float(note.start), 6),
                int(note.pitch),
                round(float(note.end), 6),
                int(inst_idx),
                int(note_idx),
            )
            refs.append((key, note))
    refs.sort(key=lambda item: item[0])
    return refs


def extract_sorted_notes(midi_path: str | Path) -> List[SortedMidiNote]:
    pm = load_pretty_midi(midi_path)
    out: List[SortedMidiNote] = []
    for key, note in _sorted_note_refs(pm):
        out.append(
            SortedMidiNote(
                onset=float(note.start),
                offset=float(note.end),
                pitch=int(note.pitch),
                velocity=int(note.velocity),
                instrument_index=int(key[3]),
                note_index=int(key[4]),
            )
        )
    return out


def replace_note_velocities(
    in_midi: str | Path,
    velocities: Sequence[int] | np.ndarray,
    out_midi: str | Path,
) -> Path:
    pm = load_pretty_midi(in_midi)
    refs = _sorted_note_refs(pm)
    vel = np.asarray(velocities, dtype=np.float64).reshape(-1)
    if vel.size != len(refs):
        raise ValueError(
            f"Velocity count mismatch: got {vel.size}, expected {len(refs)} notes from {in_midi}"
        )
    for (_, note), value in zip(refs, vel):
        note.velocity = int(np.clip(round(float(value)), 1, 127))

    out_midi = Path(out_midi)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi))
    return out_midi


def write_flat_velocity_copy(
    in_midi: str | Path,
    out_midi: str | Path,
    *,
    flat_velocity: int = 64,
) -> Path:
    notes = extract_sorted_notes(in_midi)
    velocities = np.full(len(notes), int(np.clip(round(flat_velocity), 1, 127)), dtype=np.int64)
    return replace_note_velocities(in_midi, velocities, out_midi)


def _direct_order_match(
    gt_notes: Sequence[SortedMidiNote],
    pred_notes: Sequence[SortedMidiNote],
    *,
    onset_tolerance_s: float,
    offset_tolerance_s: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(gt_notes) != len(pred_notes):
        return None
    gt_vel = np.zeros(len(gt_notes), dtype=np.float64)
    pred_vel = np.zeros(len(pred_notes), dtype=np.float64)
    for idx, (gt, pred) in enumerate(zip(gt_notes, pred_notes)):
        if int(gt.pitch) != int(pred.pitch):
            return None
        if abs(float(gt.onset) - float(pred.onset)) > onset_tolerance_s:
            return None
        if abs((float(gt.offset) - float(gt.onset)) - (float(pred.offset) - float(pred.onset))) > offset_tolerance_s:
            return None
        gt_vel[idx] = float(gt.velocity)
        pred_vel[idx] = float(pred.velocity)
    return gt_vel, pred_vel


def _greedy_pitch_onset_match(
    gt_notes: Sequence[SortedMidiNote],
    pred_notes: Sequence[SortedMidiNote],
    *,
    onset_tolerance_s: float,
    offset_tolerance_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_by_pitch: Dict[int, List[int]] = {}
    for idx, note in enumerate(pred_notes):
        pred_by_pitch.setdefault(int(note.pitch), []).append(idx)

    used = np.zeros(len(pred_notes), dtype=bool)
    gt_vel: List[float] = []
    pred_vel: List[float] = []

    for gt in gt_notes:
        candidates = pred_by_pitch.get(int(gt.pitch), [])
        best_idx = None
        best_score = None
        gt_dur = float(gt.offset) - float(gt.onset)
        for idx in candidates:
            if used[idx]:
                continue
            pred = pred_notes[idx]
            onset_diff = abs(float(gt.onset) - float(pred.onset))
            if onset_diff > onset_tolerance_s:
                continue
            dur_diff = abs(gt_dur - (float(pred.offset) - float(pred.onset)))
            if dur_diff > offset_tolerance_s:
                continue
            score = (onset_diff, dur_diff, abs(float(gt.offset) - float(pred.offset)))
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            continue
        used[best_idx] = True
        gt_vel.append(float(gt.velocity))
        pred_vel.append(float(pred_notes[best_idx].velocity))

    return np.asarray(gt_vel, dtype=np.float64), np.asarray(pred_vel, dtype=np.float64)


def align_note_velocities(
    gt_midi: str | Path,
    pred_midi: str | Path,
    *,
    onset_tolerance_s: float = 0.03,
    offset_tolerance_s: float = 0.08,
) -> VelocityAlignmentResult:
    gt_notes = extract_sorted_notes(gt_midi)
    pred_notes = extract_sorted_notes(pred_midi)

    direct = _direct_order_match(
        gt_notes,
        pred_notes,
        onset_tolerance_s=onset_tolerance_s,
        offset_tolerance_s=offset_tolerance_s,
    )
    matched_in_exact_order = direct is not None
    if direct is None:
        gt_vel, pred_vel = _greedy_pitch_onset_match(
            gt_notes,
            pred_notes,
            onset_tolerance_s=onset_tolerance_s,
            offset_tolerance_s=offset_tolerance_s,
        )
    else:
        gt_vel, pred_vel = direct

    if gt_vel.size == 0 or pred_vel.size == 0:
        mae = float("nan")
        matched = 0
    else:
        matched = int(min(gt_vel.size, pred_vel.size))
        mae = float(np.mean(np.abs(gt_vel[:matched] - pred_vel[:matched])))

    return VelocityAlignmentResult(
        num_gt_notes=int(len(gt_notes)),
        num_pred_notes=int(len(pred_notes)),
        num_matched_notes=int(matched),
        mae=mae,
        matched_in_exact_order=matched_in_exact_order,
        unmatched_gt=int(max(0, len(gt_notes) - matched)),
        unmatched_pred=int(max(0, len(pred_notes) - matched)),
    )
