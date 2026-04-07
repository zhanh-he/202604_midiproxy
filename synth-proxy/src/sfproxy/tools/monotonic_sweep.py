from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch


@dataclass
class SweepConfig:
    pitch: int = 60
    onset_norm: float = 0.1
    dur_norm: float = 0.7
    nmax: int = 8

    vel_grid: int = 33
    eps: float = 1e-4


def _make_single_note_batch(cfg: SweepConfig, device: torch.device):
    pitch = torch.zeros((1, cfg.nmax), dtype=torch.long, device=device)
    cont = torch.zeros((1, cfg.nmax, 3), dtype=torch.float32, device=device)
    mask = torch.zeros((1, cfg.nmax), dtype=torch.bool, device=device)

    pitch[0, 0] = int(cfg.pitch)
    cont[0, 0, 0] = float(cfg.onset_norm)
    cont[0, 0, 1] = float(cfg.dur_norm)
    # cont[...,2] velocity filled later
    mask[0, 0] = True
    return pitch, cont, mask


@torch.no_grad()
def run_monotonic_sweep(
    model: torch.nn.Module,
    out_dir: Path,
    cfg: SweepConfig = SweepConfig(),
    feature_index: int = 0,
    device: Optional[str] = None,
) -> dict:
    """Run monotonic sweep on a proxy model.

    Args:
        model: module returning (note_out, seg_out)
        out_dir: directory to save json and optional plot
        feature_index: which note feature to test monotonicity on (default: 0 = harmonic energy)
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    pitch, cont, mask = _make_single_note_batch(cfg, dev)

    v = torch.linspace(0.0, 1.0, int(cfg.vel_grid), device=dev)
    y = []

    for vi in v:
        cont_i = cont.clone()
        cont_i[0, 0, 2] = vi
        note_out, _ = model(pitch=pitch, cont=cont_i, mask=mask)
        y.append(float(note_out[0, 0, feature_index].item()))

    # violation count
    viol = 0
    for i in range(1, len(y)):
        if y[i] + float(cfg.eps) < y[i - 1]:
            viol += 1

    stats = {
        "pitch": int(cfg.pitch),
        "vel_grid": int(cfg.vel_grid),
        "feature_index": int(feature_index),
        "values": y,
        "violations": int(viol),
        "violation_rate": float(viol / max(1, (len(y) - 1))),
    }

    (out_dir / "monotonic_sweep.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(v.cpu().numpy(), y)
        plt.xlabel("velocity")
        plt.ylabel(f"feature[{feature_index}]")
        plt.title("Monotonic sweep")
        plt.tight_layout()
        plt.savefig(out_dir / "monotonic_sweep.png", dpi=150)
        plt.close()
    except Exception:
        pass

    return stats
