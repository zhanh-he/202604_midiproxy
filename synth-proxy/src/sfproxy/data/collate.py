from __future__ import annotations

"""Collate utilities.

The exported dataset stores fixed-length padded tensors, so the default PyTorch
collate function works. This module exists for future extensions where variable-length
note sequences are used.
"""

from typing import Any, Dict, List

import torch


def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Simple dict collate."""
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
