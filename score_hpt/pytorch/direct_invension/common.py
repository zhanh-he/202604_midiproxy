from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict


def repo_root() -> Path:
    """Return the repository root (202604_midiproxy)."""
    return Path(__file__).resolve().parents[3]


def score_hpt_root() -> Path:
    return repo_root() / "score_hpt"


def ensure_repo_imports() -> None:
    """Add local repo packages to sys.path when notebooks/scripts run in-place.

    We keep imports local instead of relying on editable installs.
    """
    candidates = [
        repo_root(),
        score_hpt_root(),
        repo_root() / "data_analysis" / "src",
        repo_root() / "synth-proxy" / "src",
    ]
    for path in candidates:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def slugify(text: str, *, max_len: int = 120) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    token = token.strip("._-") or "item"
    return token[:max_len]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def dump_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
