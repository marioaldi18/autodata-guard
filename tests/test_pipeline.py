import json
import tempfile
import unittest
from pathlib import Path

from autodataguard.cli import main
from autodataguard.config import resolve_config
from autodataguard.pipeline import run_pipeline

SMALL = {"data": {"n_rows": 2000}}


class ConfigTests(unittest.TestCase):
    def test_invalid_configs_raise(self):
        for bad in ({"data": {"n_rows": 50}}, {"data": {"anomaly_rate": 1.2}},
                    {"split": {"val_size": 0.6, "test_size": 0.5}}, {"threshold": {"max_fpr": 2}}):
            with self.assertRaises(ValueError):
                resolve_config(bad)

    def test_override_keeps_other_defaults(self):
        cfg = resolve_config({"data": {"n_rows": 500}})
        self.assertEqual(cfg["data"]["n_rows"], 500)
        self.assertEqual(cfg["data"]["anomaly_rate"], 0.02)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_outputs_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            r1 = run_pipeline(SMALL, root=d1)
            r2 = run_pipeline(SMALL, root=d2)
            for name in ("report.md", "metrics.json", "data_dictionary.md"):
                self.assertTrue((Path(d1) / "reports" / name).exists(), name)
            self.assertEqual(len(list((Path(d1) / "visualizations").glob("*.png"))), 5)
            self.assertEqual(r1["models"], r2["models"])
            self.assertEqual(r1["validation"]["rule_engine_eval"]["recall"], 1.0)
            saved = json.loads((Path(d1) / "reports" / "metrics.json").read_text())
            self.assertEqual(saved["selected_model"], r1["selected_model"])

    def test_report_numbers_match_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_pipeline(SMALL, root=d)
            report = (Path(d) / "reports" / "report.md").read_text()
            sel = r["selected_model"]
            self.assertIn(f"{r['models'][sel]['test']['pr_auc']:.3f}", report)
            self.assertIn(f"{r['validation']['overall_score']:.4f}", report)

    def test_no_label_leakage_into_features(self):
        from autodataguard.features.preprocessing import FEATURE_NAMES
        self.assertFalse({"is_anomaly", "anomaly_type", "has_dq_issue", "dq_issue_type"} & set(FEATURE_NAMES))

    def test_cli_generate_and_run(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main(["generate", "--root", d]), 0)
            self.assertTrue((Path(d) / "data" / "synthetic" / "orders.csv").exists())


if __name__ == "__main__":
    unittest.main()
