# pylint: disable=W1203
"""Evaluate monotonicity of a trained note proxy."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import hydra
from omegaconf import DictConfig

import torch

from sfproxy.models.lit_module import NoteProxyLitModule
from sfproxy.tools.monotonic_sweep import SweepConfig, run_monotonic_sweep


def run_eval_monotonic(cfg: DictConfig) -> None:
    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

    # Recreate architecture from config
    model_nn = hydra.utils.instantiate(cfg.model.cfg)

    lit = NoteProxyLitModule.load_from_checkpoint(str(ckpt_path), model=model_nn)

    out_dir = Path.cwd() / "eval_monotonic"

    sweep_cfg = SweepConfig(
        pitch=int(cfg.monotonic.sweep.pitch),
        onset_norm=float(cfg.monotonic.sweep.onset_norm),
        dur_norm=float(cfg.monotonic.sweep.dur_norm),
        nmax=int(cfg.monotonic.sweep.nmax),
        vel_grid=int(cfg.monotonic.sweep.vel_grid),
        eps=float(cfg.monotonic.sweep.eps),
    )

    stats = run_monotonic_sweep(
        model=lit.model,
        out_dir=out_dir,
        cfg=sweep_cfg,
        feature_index=int(cfg.monotonic.feature_index),
        device=str(cfg.device) if cfg.get("device") else None,
    )

    print(stats)


@hydra.main(version_base="1.3", config_path="../../configs/sfproxy", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_eval_monotonic(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
