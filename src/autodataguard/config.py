"""Configuration defaults, loading and validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "data": {
        "n_rows": 10000,
        "start_date": "2024-01-01",
        "reference_date": "2025-12-31",
        "anomaly_rate": 0.02,
        "dq_issue_rate": 0.06,
        "duplicate_rate": 0.01,
    },
    "split": {"val_size": 0.2, "test_size": 0.2},
    "models": {
        "isolation_forest": {"n_estimators": 200},
        "lof": {"n_neighbors": 35},
    },
    "threshold": {"max_fpr": 0.01},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def validate_config(cfg: Dict[str, Any]) -> None:
    data, split = cfg["data"], cfg["split"]
    if int(data["n_rows"]) < 200:
        raise ValueError("data.n_rows must be at least 200")
    for key in ("anomaly_rate", "dq_issue_rate", "duplicate_rate"):
        if not 0.0 <= float(data[key]) < 1.0:
            raise ValueError(f"data.{key} must be in [0, 1)")
    total = data["anomaly_rate"] + data["dq_issue_rate"] + data["duplicate_rate"]
    if total >= 1.0:
        raise ValueError("anomaly_rate + dq_issue_rate + duplicate_rate must be < 1")
    if not (0 < split["val_size"] < 1 and 0 < split["test_size"] < 1):
        raise ValueError("split sizes must be in (0, 1)")
    if split["val_size"] + split["test_size"] >= 1:
        raise ValueError("val_size + test_size must be < 1")
    if not 0.0 <= cfg["threshold"]["max_fpr"] <= 1.0:
        raise ValueError("threshold.max_fpr must be in [0, 1]")


def resolve_config(override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = _deep_merge(DEFAULT_CONFIG, override or {})
    validate_config(cfg)
    return cfg


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path is None:
        return resolve_config()
    with open(path, "r", encoding="utf-8") as fh:
        return resolve_config(json.load(fh))
