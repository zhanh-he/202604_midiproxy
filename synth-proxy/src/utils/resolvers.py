"""Resolvers for perceptual loss scaling and losses schedulers in hydra config"""

import math

from omegaconf import OmegaConf


def resolve_mul(x: float, y: float) -> float:
    return x * y


def resolve_start(total_num_steps: int, start_ratio: float) -> float:
    return int(start_ratio * total_num_steps)


def resolve_warmp(total_num_steps: int, warmup_ratio: float) -> float:
    return int(warmup_ratio * total_num_steps)


def resolve_total_num_steps(num_epochs: int, dataset_size: int, batch_size: int) -> float:
    return num_epochs * dataset_size // batch_size


def resolve_div_int(x: float, y: float) -> int:
    return int(round(float(x) / float(y)))


def resolve_hop_from_sr_fps(sample_rate: float, frame_rate: float) -> int:
    return int(math.floor(float(sample_rate) / float(frame_rate) + 0.5))


OmegaConf.register_new_resolver("mul", resolve_mul)
OmegaConf.register_new_resolver("start", resolve_start)
OmegaConf.register_new_resolver("warm", resolve_warmp)
OmegaConf.register_new_resolver("total_num_steps", resolve_total_num_steps)

OmegaConf.register_new_resolver("div_int", resolve_div_int)
OmegaConf.register_new_resolver("hop_from_sr_fps", resolve_hop_from_sr_fps)
