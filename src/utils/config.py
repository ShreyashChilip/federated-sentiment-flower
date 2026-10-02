"""YAML configuration loading with inheritance and dotted overrides."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parents = cfg.pop("inherits", [])
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = deep_merge(merged, _load_file(path.parent / parent))
    return deep_merge(merged, cfg)


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    node = cfg
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(*names: str, overrides: dict | None = None) -> dict:
    """Merge one or more config files (later ones win), then apply overrides.

    ``names`` are file names inside ``configs/`` or explicit paths.
    ``overrides`` maps dotted keys (``"fl.mu"``) to values.
    """
    cfg: dict = {}
    for name in names:
        path = Path(name)
        if not path.exists():
            path = CONFIG_DIR / name
        cfg = deep_merge(cfg, _load_file(path))
        if cfg.get("not_implemented"):
            raise NotImplementedError(f"{path.name} is a placeholder; read the comment at the top of the file")
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    return cfg


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key=value`` from the command line; the value is read as YAML."""
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)


def config_hash(cfg: dict, length: int = 12) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]
