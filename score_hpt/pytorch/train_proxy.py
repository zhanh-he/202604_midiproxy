from __future__ import annotations

import sys

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from train_backend import train as train_backend

def _compose_cfg(default_overrides):
    GlobalHydra.instance().clear()
    initialize(config_path="./config", job_name="train_proxy", version_base=None)
    return compose(config_name="config", overrides=list(default_overrides) + sys.argv[1:])


if __name__ == "__main__":
    cfg = _compose_cfg([
        "proxy.enabled=true",
        "proxy.type=diffproxy",
        "loss.proxy_weight=1.0",
        "proxy.backend_segment_seconds=0.0",
    ])
    train_backend(cfg)
