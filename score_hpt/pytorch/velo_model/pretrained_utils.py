from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


def resolve_pretrained_checkpoint(raw_path: str, model_label: str, required: bool) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        if required:
            raise ValueError(f"{model_label} requires model.pretrained_checkpoint to be set.")
        return None

    checkpoint_path = Path(text).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{model_label} pretrained checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def unwrap_checkpoint_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint format: expected a state dict or a dict containing one.")

    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def select_prefixed_substate(
    state_dict: Dict[str, torch.Tensor],
    model_keys: Iterable[str],
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    required_keys = list(model_keys)
    required_set = set(required_keys)

    for prefix in prefixes:
        if prefix:
            candidate = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        else:
            candidate = dict(state_dict)

        if required_set.issubset(candidate.keys()):
            return {key: candidate[key] for key in required_keys}

    raise RuntimeError(
        "Checkpoint does not contain a compatible model state dict under any supported prefix: "
        f"{list(prefixes)}"
    )
