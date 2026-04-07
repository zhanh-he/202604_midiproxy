from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import hydra
from omegaconf import DictConfig

from sfproxy.eval_monotonic import run_eval_monotonic
from sfproxy.eval_velocity_recovery import run_eval_velocity_recovery


@hydra.main(version_base="1.3", config_path="../../configs/sfproxy", config_name="eval")
def main(cfg: DictConfig) -> None:
    mode = str(cfg.mode or "monotonic").strip().lower()
    if mode == "monotonic":
        run_eval_monotonic(cfg)
        return
    if mode == "velocity_recovery":
        run_eval_velocity_recovery(cfg)
        return
    raise ValueError(f"Unsupported eval mode: {mode}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
