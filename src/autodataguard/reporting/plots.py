"""Reproducible static visualisations (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dimension_scores(scores: Dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names, vals = list(scores), list(scores.values())
    ax.bar(names, vals, color="#3b6ea5")
    ax.set_ylim(min(0.9, min(vals) - 0.01), 1.0)
    ax.set_ylabel("score")
    ax.set_title("Data quality dimension scores")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _save(fig, path)


def plot_violations_by_rule(counts: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if counts.empty:
        ax.text(0.5, 0.5, "No violations", ha="center", va="center")
    else:
        d = counts.sort_values("violations")
        ax.barh(d["rule"], d["violations"], color="#c0504d")
        ax.set_xlabel("violations")
    ax.set_title("Rule violations by rule")
    _save(fig, path)


def plot_pr_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], prevalence: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=name)
    ax.axhline(prevalence, color="grey", linestyle="--", label=f"prevalence ({prevalence:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall curves (test split)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_score_distribution(y: np.ndarray, s: np.ndarray, threshold: float, model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(s.min(), s.max(), 50)
    ax.hist(s[y == 0], bins=bins, alpha=0.7, label="normal", color="#3b6ea5")
    ax.hist(s[y == 1], bins=bins, alpha=0.8, label="injected anomaly", color="#c0504d")
    ax.axvline(threshold, color="black", linestyle="--", label="threshold (max F1 on validation)")
    ax.set_yscale("log")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("rows (log scale)")
    ax.set_title(f"Score distribution: {model} (test split)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_recall_by_type(per_type: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    d = per_type.dropna(subset=["recall"])
    ax.bar(d["anomaly_type"], d["recall"], color="#4f9a5b")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall at selected threshold")
    ax.set_title("Recall by injected anomaly type (test split)")
    for i, (r, n) in enumerate(zip(d["recall"], d["n_test"])):
        ax.text(i, r, f"{r:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelrotation=15)
    _save(fig, path)
