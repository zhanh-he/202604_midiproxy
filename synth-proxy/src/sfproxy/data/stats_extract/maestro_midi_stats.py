from __future__ import annotations

"""Extract note statistics from MAESTRO-style MIDI files.

This utility outputs a json with empirical samples:
- pitches
- durations (seconds)
- iois (seconds)
- chord_sizes

The script is intentionally simple and can be adapted per dataset.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def extract_from_midi(path: Path, chord_window_s: float = 0.02) -> Dict:
    try:
        import mido  # type: ignore
    except Exception as e:
        raise RuntimeError("mido is required to parse MIDI") from e

    mid = mido.MidiFile(str(path))
    ticks_per_beat = mid.ticks_per_beat

    tempo = 500000  # default 120 BPM

    # Track note onsets and offsets
    on_times: Dict[Tuple[int, int], float] = {}
    onsets: List[float] = []
    pitches: List[int] = []
    durations: List[float] = []

    time_s = 0.0

    for msg in mido.merge_tracks(mid.tracks):
        time_s += float(mido.tick2second(msg.time, ticks_per_beat, tempo))
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            key = (int(msg.channel), int(msg.note))
            on_times[key] = time_s
            onsets.append(time_s)
            pitches.append(int(msg.note))

        if (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
            key = (int(msg.channel), int(msg.note))
            if key in on_times:
                t0 = on_times.pop(key)
                durations.append(max(0.0, time_s - t0))

    # IOI
    onsets_sorted = sorted(onsets)
    iois = [max(0.0, onsets_sorted[i] - onsets_sorted[i - 1]) for i in range(1, len(onsets_sorted))]

    # chord sizes (cluster onsets)
    chord_sizes: List[int] = []
    i = 0
    while i < len(onsets_sorted):
        j = i + 1
        while j < len(onsets_sorted) and (onsets_sorted[j] - onsets_sorted[i]) <= chord_window_s:
            j += 1
        chord_sizes.append(j - i)
        i = j

    return {
        "pitches": pitches,
        "durations": durations,
        "iois": iois,
        "chord_sizes": chord_sizes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    midi_root = Path(args.midi_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(midi_root.rglob("*.mid")) + sorted(midi_root.rglob("*.midi"))
    if args.limit and args.limit > 0:
        midi_files = midi_files[: args.limit]

    agg = {"pitches": [], "durations": [], "iois": [], "chord_sizes": []}

    for p in midi_files:
        s = extract_from_midi(p)
        for k in agg:
            agg[k].extend(s[k])

    out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"Wrote stats to {out_path} from {len(midi_files)} MIDI files")


if __name__ == "__main__":
    main()
