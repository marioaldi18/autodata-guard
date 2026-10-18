"""Command-line interface: ``python -m autodataguard run --config configs/default.json``."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import load_config
from .data.generator import generate_orders
from .pipeline import run_pipeline


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="autodataguard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (("run", "run the full pipeline and write reports"),
                        ("generate", "only generate the synthetic dataset")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", default=None, help="path to a JSON config (defaults used if omitted)")
        p.add_argument("--root", default=".", help="project root where data/, reports/ are written")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "generate":
        from pathlib import Path
        out = Path(args.root) / "data" / "synthetic"
        out.mkdir(parents=True, exist_ok=True)
        raw, labels = generate_orders(cfg["data"], cfg["seed"])
        raw.to_csv(out / "orders.csv", index=False)
        labels.to_csv(out / "orders_labels.csv", index=False)
        print(f"Wrote {len(raw)} rows to {out}")
        return 0

    res = run_pipeline(cfg, root=args.root)
    sel = res["selected_model"]
    t = res["models"][sel]["test"]
    print(f"Selected detector: {sel}")
    print(f"Test PR-AUC {t['pr_auc']:.3f} | ROC-AUC {t['roc_auc']:.3f} | "
          f"F1 {t['at_max_f1']['f1']:.3f} | data-quality score {res['validation']['overall_score']:.4f}")
    print("Report: reports/report.md")
    return 0
