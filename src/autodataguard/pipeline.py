"""End-to-end pipeline: generate -> validate -> preprocess -> detect -> evaluate -> report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import resolve_config
from .data.generator import generate_orders
from .data.io import load_raw
from .data.validation import coerce_types, validate
from .evaluation.error_analysis import evaluate_rule_engine, false_positive_analysis, per_type_recall
from .evaluation.metrics import (ranking_metrics, select_threshold_fpr_budget, select_threshold_max_f1,
                                 threshold_metrics)
from .evaluation.splits import split_indices
from .features.preprocessing import FeaturePreprocessor
from .models.detectors import build_detectors
from .reporting import plots
from .reporting.dictionary import render_data_dictionary
from .reporting.report import render_report


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.ndarray)):
        return str(o) if isinstance(o, pd.Timestamp) else o.tolist()
    raise TypeError(f"Cannot serialise {type(o)}")


def run_pipeline(config: Optional[Dict[str, Any]] = None, root: str | Path = ".") -> Dict[str, Any]:
    cfg = resolve_config(config)
    root = Path(root)
    seed = cfg["seed"]
    data_dir, reports_dir, viz_dir = root / "data" / "synthetic", root / "reports", root / "visualizations"
    for d in (data_dir, reports_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic data (written to disk, then re-read to exercise ingestion)
    raw, labels = generate_orders(cfg["data"], seed)
    raw.to_csv(data_dir / "orders.csv", index=False)
    labels.to_csv(data_dir / "orders_labels.csv", index=False)
    raw_loaded = load_raw(data_dir / "orders.csv")

    # 2. Rule-based validation
    validation = validate(raw_loaded, reference_date=cfg["data"]["reference_date"])
    rule_eval = evaluate_rule_engine(validation, labels)

    # 3. Modelling set: typed values, exact duplicate rows removed
    typed = coerce_types(raw_loaded)
    dup_rows = validation.rows_for_rule("duplicate_key")
    model_typed = typed.drop(index=dup_rows)
    model_labels = labels.set_index("row_index").loc[model_typed.index]
    y_all = model_labels["is_anomaly"].astype(int)

    # 4. Stratified split; preprocessing fitted on train only
    splits = split_indices(y_all, cfg["split"]["val_size"], cfg["split"]["test_size"], seed)
    pre = FeaturePreprocessor().fit(model_typed.loc[splits["train"]])
    X = {k: pre.transform(model_typed.loc[idx]) for k, idx in splits.items()}
    y = {k: y_all.loc[idx].to_numpy() for k, idx in splits.items()}
    for k in ("val", "test"):
        if y[k].sum() < 1:
            raise ValueError(f"The {k} split has no anomalies; increase n_rows or anomaly_rate")

    # 5. Fit detectors, choose thresholds on validation, evaluate on test
    models: Dict[str, Any] = {}
    scores: Dict[str, Dict[str, np.ndarray]] = {}
    for det in build_detectors(cfg):
        det.fit(X["train"].to_numpy())
        s_val, s_test = det.score(X["val"].to_numpy()), det.score(X["test"].to_numpy())
        scores[det.name] = {"val": s_val, "test": s_test}
        thr_f1 = select_threshold_max_f1(y["val"], s_val)
        thr_fpr = select_threshold_fpr_budget(y["val"], s_val, cfg["threshold"]["max_fpr"])
        models[det.name] = {
            "val": ranking_metrics(y["val"], s_val),
            "val_at_max_f1": threshold_metrics(y["val"], s_val, thr_f1),
            "test": {**ranking_metrics(y["test"], s_test),
                     "at_max_f1": threshold_metrics(y["test"], s_test, thr_f1),
                     "at_fpr_budget": threshold_metrics(y["test"], s_test, thr_fpr)},
        }
    selected = max(models, key=lambda n: models[n]["val"]["pr_auc"])
    thr = models[selected]["test"]["at_max_f1"]["threshold"]
    flagged_test = scores[selected]["test"] >= thr
    lab_test = model_labels.loc[splits["test"]]
    per_type = per_type_recall(lab_test, flagged_test)
    fp = false_positive_analysis(X["test"], model_typed, model_labels, scores[selected]["test"],
                                 y["test"], thr)

    results: Dict[str, Any] = {
        "config": cfg,
        "dataset": {
            "n_rows_raw": len(raw_loaded), "n_rows_modeling": len(model_typed),
            "n_duplicates_removed": len(dup_rows), "n_anomalies": int(y_all.sum()),
            "anomaly_rate_modeling": float(y_all.mean()),
            "splits": [{"split": k, "rows": len(splits[k]), "anomalies": int(y[k].sum())} for k in splits],
        },
        "validation": {
            "dimension_scores": validation.dimension_scores, "overall_score": validation.overall_score,
            "counts_by_rule": validation.counts_by_rule().to_dict("records"), "rule_engine_eval": rule_eval,
        },
        "models": models, "selected_model": selected, "selection_criterion": "validation PR-AUC",
        "per_type_recall": per_type, "false_positive_analysis": fp,
    }

    # 6. Artefacts
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    (reports_dir / "data_dictionary.md").write_text(render_data_dictionary(), encoding="utf-8")
    (reports_dir / "report.md").write_text(render_report(results), encoding="utf-8")
    plots.plot_dimension_scores(validation.dimension_scores, viz_dir / "dq_dimension_scores.png")
    plots.plot_violations_by_rule(validation.counts_by_rule(), viz_dir / "dq_violations_by_rule.png")
    plots.plot_pr_curves({n: (y["test"], scores[n]["test"]) for n in scores},
                         float(y["test"].mean()), viz_dir / "pr_curves.png")
    plots.plot_score_distribution(y["test"], scores[selected]["test"], thr, selected,
                                  viz_dir / "score_distribution.png")
    plots.plot_recall_by_type(pd.DataFrame(per_type), viz_dir / "recall_by_type.png")
    return results
