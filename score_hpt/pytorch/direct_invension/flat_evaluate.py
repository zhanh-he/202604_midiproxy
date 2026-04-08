from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.eval_runner import run_evaluation
from direct_invension.common import compose_cfg, normalize_dataset_type


def main() -> None:
    from omegaconf import OmegaConf

    cfg = compose_cfg(sys.argv[1:], job_name="flat_evaluate")
    dataset_type = normalize_dataset_type(cfg.dataset.test_set)
    flat_velocity = int(cfg.flat.infer.flat_velocity)

    result = run_evaluation(
        cfg=cfg,
        dataset_type=dataset_type,
        eval_cfg=cfg.flat.eval,
        config_prefix="flat.eval",
        route_name="Flat velocity eval",
        run_json_name="flat_eval_run.json",
        extra_run_payload={
            "flat_velocity": flat_velocity,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        extra_summary_lines=[f"  flat_velocity: {flat_velocity}"],
    )
    print(result["summary_text"])


if __name__ == "__main__":
    main()
