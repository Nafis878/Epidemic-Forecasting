"""Loader for ``configs/experiment.yaml``.

A thin, dependency-light accessor so the run profiles, hybrid hyper-parameters,
and bootstrap settings have one documented home. Callers fall back to their own
embedded defaults if the file or PyYAML is unavailable, so the pipeline never
hard-fails on a missing config.
"""

from __future__ import annotations

import os
from typing import Optional

HERE = os.path.dirname(__file__)
CONFIG_PATH = os.path.abspath(os.path.join(HERE, "..", "configs", "experiment.yaml"))


def load_experiment(path: str = CONFIG_PATH) -> Optional[dict]:
    """Return the parsed experiment config, or ``None`` if unavailable."""
    try:
        import yaml  # local import: optional dependency
    except ImportError:
        return None
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def get_profile(name: str, fallback: Optional[dict] = None) -> Optional[dict]:
    """Return a single run profile (``quick``/``full``) from the config or fallback."""
    cfg = load_experiment()
    if cfg and "profiles" in cfg and name in cfg["profiles"]:
        return cfg["profiles"][name]
    return fallback
