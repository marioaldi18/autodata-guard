import unittest

import pandas as pd

from autodataguard.config import resolve_config
from autodataguard.data.generator import ANOMALY_TYPES, generate_orders
from autodataguard.data.schema import COLUMN_NAMES
from autodataguard.data.validation import validate


def small_cfg(**data):
    return resolve_config({"data": {"n_rows": 1000, **data}})["data"]


class GeneratorTests(unittest.TestCase):
    def test_same_seed_is_identical(self):
        a, la = generate_orders(small_cfg(), 7)
        b, lb = generate_orders(small_cfg(), 7)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(la, lb)

    def test_different_seed_differs(self):
        a, _ = generate_orders(small_cfg(), 1)
        b, _ = generate_orders(small_cfg(), 2)
        self.assertFalse(a.equals(b))

    def test_columns_and_label_alignment(self):
        raw, labels = generate_orders(small_cfg(), 3)
        self.assertEqual(list(raw.columns), COLUMN_NAMES)
        self.assertEqual(len(raw), len(labels))
        self.assertNotIn("is_anomaly", raw.columns)  # labels never ship with features

    def test_injected_counts(self):
        raw, labels = generate_orders(small_cfg(), 3)
        self.assertEqual(int(labels["is_anomaly"].sum()), 20)
        self.assertEqual(int((labels["dq_issue_type"] == "duplicate_key").sum()), 10)
        self.assertEqual(len(raw), 1010)
        self.assertEqual(set(labels.loc[labels.is_anomaly, "anomaly_type"]), set(ANOMALY_TYPES))

    def test_anomalies_and_dq_rows_do_not_overlap(self):
        _, labels = generate_orders(small_cfg(), 3)
        self.assertFalse((labels["is_anomaly"] & labels["has_dq_issue"]).any())

    def test_clean_config_passes_validation(self):
        raw, labels = generate_orders(small_cfg(dq_issue_rate=0.0, duplicate_rate=0.0, anomaly_rate=0.0), 5)
        self.assertEqual(len(validate(raw).violations), 0)
        self.assertFalse(labels["has_dq_issue"].any())

    def test_anomalies_pass_rule_validation(self):
        raw, _ = generate_orders(small_cfg(dq_issue_rate=0.0, duplicate_rate=0.0), 5)
        self.assertEqual(len(validate(raw).violations), 0)  # anomalies are statistical, not rule violations

    def test_too_many_injections_raises(self):
        with self.assertRaises(ValueError):
            generate_orders({**small_cfg(), "anomaly_rate": 0.5, "dq_issue_rate": 0.4, "duplicate_rate": 0.3}, 1)


if __name__ == "__main__":
    unittest.main()
