"""YAML config loading: ~ and ${ENV} expansion, ${self.path} references, CLI overrides (a.b.c=value)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_REF = re.compile(r"\$\{([^}]+)\}")


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def _expand(x: Any, root: dict, base: Path):
    if isinstance(x, dict):
        return {k: _expand(v, root, base) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v, root, base) for v in x]
    if isinstance(x, str):
        def sub(m):
            key = m.group(1)
            if key.startswith("self."):
                return str(_get(root, key[5:]))
            if key == "config_dir":
                return str(base)
            return os.environ.get(key, m.group(0))
        s = _REF.sub(sub, x)
        if s.startswith("~"):
            s = os.path.expanduser(s)
        return s
    return x


def set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {}) if isinstance(d, dict) else d[int(k)]
    if isinstance(d, list):
        d[int(keys[-1])] = value
    else:
        d[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        set_dotted(cfg, k, yaml.safe_load(v))
    cfg = _expand(cfg, cfg, path.parent)
    cfg.setdefault("name", path.stem)
    cfg["_config_path"] = str(path)
    return cfg
