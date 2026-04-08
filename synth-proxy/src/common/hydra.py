"""Hydra helpers used by the sfproxy entrypoints."""

from __future__ import annotations

import math
from typing import Callable, List

import hydra
from lightning import Callback
from omegaconf import DictConfig, OmegaConf

from common.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _register_resolver(name: str, fn: Callable) -> None:
    if OmegaConf.has_resolver(name):
        return
    OmegaConf.register_new_resolver(name, fn)


def _resolve_seg_tag(seg_len_s: float) -> str:
    value = float(seg_len_s)
    if value.is_integer():
        return f"{int(value)}s"
    return f"{value:g}".replace(".", "p") + "s"


def _resolve_split_value(split: str, train_value, val_value):
    split_name = str(split).strip().lower()
    if split_name == "train":
        return train_value
    if split_name == "val":
        return val_value
    raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'val'.")


def register_resolvers() -> None:
    _register_resolver("mul", lambda x, y: x * y)
    _register_resolver("start", lambda total_num_steps, start_ratio: int(start_ratio * total_num_steps))
    _register_resolver("warm", lambda total_num_steps, warmup_ratio: int(warmup_ratio * total_num_steps))
    _register_resolver(
        "total_num_steps",
        lambda num_epochs, dataset_size, batch_size: num_epochs * dataset_size // batch_size,
    )
    _register_resolver("div_int", lambda x, y: int(round(float(x) / float(y))))
    _register_resolver(
        "hop_from_sr_fps",
        lambda sample_rate, frame_rate: int(math.floor(float(sample_rate) / float(frame_rate) + 0.5)),
    )
    _register_resolver("seg_tag", _resolve_seg_tag)
    _register_resolver("split_value", _resolve_split_value)


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> List[Callback] | None:
    """Instantiate Lightning callbacks from a Hydra config."""
    if not callbacks_cfg:
        log.info("No callback configs found. Skipping...")
        return None
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig")

    callbacks: List[Callback] = []
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


register_resolvers()
