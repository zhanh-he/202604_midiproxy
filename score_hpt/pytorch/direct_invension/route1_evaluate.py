from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.eval_runner import run_evaluation
from direct_invension.common import compose_cfg, normalize_dataset_type


def main() -> None:
    from omegaconf import OmegaConf

    cfg = compose_cfg(sys.argv[1:], job_name="route1_evaluate")
    dataset_type = normalize_dataset_type(cfg.dataset.test_set)

    result = run_evaluation(
        cfg=cfg,
        dataset_type=dataset_type,
        eval_cfg=cfg.route1.eval,
        config_prefix="route1.eval",
        route_name="Route I eval",
        run_json_name="route1_eval_run.json",
        extra_run_payload={"config": OmegaConf.to_container(cfg, resolve=True)},
    )
    print(result["summary_text"])


if __name__ == "__main__":
    main()
