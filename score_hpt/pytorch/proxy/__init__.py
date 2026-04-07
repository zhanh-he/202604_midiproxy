"""Differentiable supervision backends for weakly supervised velocity training.

Historical note:
The python package name `proxy` is kept for backward compatibility.
In project docs we use these terms instead:
- DiffSynth = DDSP-based differentiable synthesizer
- DiffProxy = differentiable proxy of a black-box synthesizer
"""

from .objective import build_proxy_objective

__all__ = ["build_proxy_objective"]
