from __future__ import annotations

"""Extract note statistics from GuitarSet JAMS files.

This is a best-effort parser. GuitarSet JAMS schemas can vary by version.
You may need to adapt the field names to your local dataset.

Output json contains empirical samples:
- pitches
- durations
- iois
- chord_sizes
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _try_get_pitch(value) -> int | None:
    if value is None:
        return None
    # value can be float MIDI
    if isinstance(value, (int, float)):
        p = int(round(float(value)))
        return p if 0 <= p <= 127 else None
    if isinstance(value, dict):
        for k in ["midi", "pitch", "pitch_midi", "note_midi"]:
            if k in value:
                return _try_get_pitch(value[k])
    return None


def extract_from_jams(path: Path, chord_window_s: float = 0.02) -> Dict:
    j = json.loads(path.read_text(encoding="utf-8"))

    pitches: List[int] = []
    durations: List[float] = []
    onsets: List[float] = []

    annotations = j.get("annotations", [])
    for ann in annotations:
        data = ann.get("data")
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            t = item.get("time")
            d = item.get("duration")
            v = item.get("value")
            if t is None or d is None:
                continue
            p = _try_get_pitch(v)
            if p is None:
                continue
            onsets.append(float(t))
            durations.append(max(0.0, float(d)))
            pitches.append(int(p))

    onsets_sorted = sorted(onsets)
    iois = [max(0.0, onsets_sorted[i] - onsets_sorted[i - 1]) for i in range(1, len(onsets_sorted))]

    chord_sizes: List[int] = []
    i = 0
    while i < len(onsets_sorted):
        j2 = i + 1
        while j2 < len(onsets_sorted) and (onsets_sorted[j2] - onsets_sorted[i]) <= chord_window_s:
            j2 += 1
        chord_sizes.append(j2 - i)
        i = j2

    return {"pitches": pitches, "durations": durations, "iois": iois, "chord_sizes": chord_sizes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jams_root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    jams_root = Path(args.jams_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jams_files = sorted(jams_root.rglob("*.jams"))
    if args.limit and args.limit > 0:
        jams_files = jams_files[: args.limit]

    agg = {"pitches": [], "durations": [], "iois": [], "chord_sizes": []}
    for p in jams_files:
        s = extract_from_jams(p)
        for k in agg:
            agg[k].extend(s[k])

    out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"Wrote stats to {out_path} from {len(jams_files)} JAMS files")


if __name__ == "__main__":
    main()
