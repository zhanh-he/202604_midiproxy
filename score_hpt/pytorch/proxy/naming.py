from __future__ import annotations

from typing import Dict


_CANONICAL_TO_DISPLAY: Dict[str, str] = {
    "ddsp_piano": "DiffSynth-Piano",
    "ddsp_guitar": "DiffSynth-Guitar",
    "sfproxy": "DiffProxy",
}

_CANONICAL_TO_RUN_TAG: Dict[str, str] = {
    "ddsp_piano": "diffsynth_piano",
    "ddsp_guitar": "diffsynth_guitar",
    "sfproxy": "diffproxy",
}


def normalize_backend_type(raw_backend_type: str | None) -> str:
    """Map historical backend names and new aliases to one canonical backend id."""
    name = str(raw_backend_type or "ddsp_piano").strip().lower()
    name = name.replace('-', '_').replace('/', '_')

    if name in {"ddsp_piano", "diffsynth_piano", "diffsynthpiano"}:
        return "ddsp_piano"
    if name in {"ddsp_guitar", "ddsp_guitar_synth", "diffsynth_guitar", "diffsynthguitar", "diffsynth_guitar_synth", "diffsynthguitarsynth"}:
        return "ddsp_guitar"
    if name in {
        "sfproxy",
        "sf_proxy",
        "sfproxy_piano",
        "sfproxy_guitar",
        "diffproxy",
        "diffproxy_piano",
        "diffproxy_guitar",
    }:
        return "sfproxy"

    if "guitar" in name and ("ddsp" in name or "diffsynth" in name):
        return "ddsp_guitar"
    if "piano" in name and ("ddsp" in name or "diffsynth" in name):
        return "ddsp_piano"
    if name.startswith("sfproxy") or name.startswith("sf_proxy") or name.startswith("diffproxy"):
        return "sfproxy"
    return name


def backend_display_name(raw_backend_type: str | None) -> str:
    canonical = normalize_backend_type(raw_backend_type)
    return _CANONICAL_TO_DISPLAY.get(canonical, str(raw_backend_type or canonical))


def backend_run_tag(raw_backend_type: str | None) -> str:
    canonical = normalize_backend_type(raw_backend_type)
    return _CANONICAL_TO_RUN_TAG.get(canonical, canonical)


def is_diffproxy_backend(raw_backend_type: str | None) -> bool:
    return normalize_backend_type(raw_backend_type) == "sfproxy"


def is_diffsynth_backend(raw_backend_type: str | None) -> bool:
    return normalize_backend_type(raw_backend_type) in {"ddsp_piano", "ddsp_guitar"}
