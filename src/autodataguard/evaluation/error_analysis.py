"""Rule-engine evaluation against injected ground truth, per-type recall and false-positive analysis."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..data.generator import ANOMALY_TYPES
from ..data.validation import ValidationResult


def _num(v: Any) -> Any:
    if pd.isna(v):
        return None
    return int(v) if float(v).is_integer() else float(v)


def evaluate_rule_engine(validation: ValidationResult, labels: pd.DataFrame) -> Dict[str, Any]:
    flagged = set(validation.flagged_rows())
    truth = set(labels.loc[labels["has_dq_issue"], "row_index"].tolist())
    tp = len(flagged & truth)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(truth) if truth else 0.0
    per_type = []
    for t, grp in labels[labels["has_dq_issue"]].groupby("dq_issue_type"):
        rows = set(grp["row_index"].tolist())
        per_type.append({"dq_issue_type": t, "n_injected": len(rows), "detected": len(rows & flagged),
                         "recall": len(rows & flagged) / len(rows)})
    return {"n_flagged_rows": len(flagged), "n_injected_rows": len(truth), "true_positive_rows": tp,
            "false_positive_rows": len(flagged - truth), "precision": precision, "recall": recall,
            "per_type": per_type}


def per_type_recall(labels_test: pd.DataFrame, flagged: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    types = labels_test["anomaly_type"].to_numpy()
    for t in ANOMALY_TYPES:
        m = types == t
        n = int(m.sum())
        out.append({"anomaly_type": t, "n_test": n, "detected": int(flagged[m].sum()),
                    "recall": float(flagged[m].mean()) if n else None})
    return out


def false_positive_analysis(features: pd.DataFrame, typed: pd.DataFrame, labels: pd.DataFrame,
                            scores: np.ndarray, y: np.ndarray, threshold: float,
                            n_examples: int = 5) -> Dict[str, Any]:
    flagged = scores >= threshold
    fp_mask = flagged & (y == 0)
    n_neg = int((y == 0).sum())
    n_fp = int(fp_mask.sum())
    result: Dict[str, Any] = {"n_false_positives": n_fp, "n_negatives": n_neg,
                              "fp_rate": n_fp / n_neg if n_neg else 0.0,
                              "share_with_dq_issue": 0.0, "top_driver_counts": {}, "examples": []}
    if n_fp == 0:
        return result
    fp_idx = features.index[fp_mask]
    fp_scores = pd.Series(scores[fp_mask], index=fp_idx)
    drivers = features.loc[fp_idx].abs().idxmax(axis=1)
    has_dq = labels.loc[fp_idx, "has_dq_issue"].astype(bool)
    result["share_with_dq_issue"] = float(has_dq.mean())
    result["top_driver_counts"] = {k: int(v) for k, v in drivers.value_counts().items()}
    top = fp_scores.sort_values(ascending=False).index[:n_examples]
    for i in top:
        row = typed.loc[i]
        result["examples"].append({
            "row_index": int(i), "score": float(fp_scores[i]), "top_driver": drivers[i],
            "product_category": row["product_category"], "quantity": _num(row["quantity"]),
            "unit_price": _num(row["unit_price"]), "discount_pct": _num(row["discount_pct"]),
            "shipping_days": _num(row["shipping_days"]), "customer_age": _num(row["customer_age"]),
            "dq_issue_type": labels.loc[i, "dq_issue_type"] or "none",
        })
    return result
