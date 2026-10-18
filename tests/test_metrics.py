import unittest

import numpy as np

from autodataguard.evaluation.metrics import (ranking_metrics, select_threshold_fpr_budget,
                                              select_threshold_max_f1, threshold_metrics)
from autodataguard.evaluation.splits import split_indices

import pandas as pd


class MetricTests(unittest.TestCase):
    def test_perfect_ranking(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        m = ranking_metrics(y, s)
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertEqual(m["pr_auc"], 1.0)
        self.assertAlmostEqual(m["prevalence"], 0.4)

    def test_single_class_raises(self):
        with self.assertRaises(ValueError):
            ranking_metrics([0, 0, 0], [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            select_threshold_max_f1([1, 1], [0.1, 0.2])

    def test_non_finite_and_shape_errors(self):
        with self.assertRaises(ValueError):
            ranking_metrics([0, 1], [np.nan, 1.0])
        with self.assertRaises(ValueError):
            ranking_metrics([0, 1], [0.1])

    def test_threshold_metrics_counts(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.6, 0.5, 0.9])
        m = threshold_metrics(y, s, 0.5)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 0, 1))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertEqual(m["recall"], 1.0)

    def test_threshold_metrics_nothing_flagged(self):
        m = threshold_metrics([0, 1], [0.1, 0.2], 10.0)
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))

    def test_max_f1_threshold_separates_clean_case(self):
        y = np.array([0] * 8 + [1] * 2)
        s = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.12, 0.22, 0.8, 0.9])
        thr = select_threshold_max_f1(y, s)
        self.assertEqual(threshold_metrics(y, s, thr)["f1"], 1.0)

    def test_fpr_budget_is_respected(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 1000 + [1] * 20)
        s = np.concatenate([rng.normal(0, 1, 1000), rng.normal(3, 1, 20)])
        for budget in (0.0, 0.01, 0.05, 1.0):
            thr = select_threshold_fpr_budget(y, s, budget)
            self.assertLessEqual(threshold_metrics(y, s, thr)["fpr"], budget + 1e-12)

    def test_fpr_budget_validation(self):
        with self.assertRaises(ValueError):
            select_threshold_fpr_budget([0, 1], [0.1, 0.2], 1.5)
        with self.assertRaises(ValueError):
            select_threshold_fpr_budget([1, 1], [0.1, 0.2], 0.1)


class SplitTests(unittest.TestCase):
    def test_stratified_disjoint_and_reproducible(self):
        y = pd.Series([1] * 50 + [0] * 950, index=np.arange(1000) + 5)
        a = split_indices(y, 0.2, 0.2, 1)
        b = split_indices(y, 0.2, 0.2, 1)
        self.assertEqual(set(a["train"]) & set(a["val"]), set())
        self.assertEqual(set(a["val"]) & set(a["test"]), set())
        self.assertEqual(len(a["train"]) + len(a["val"]) + len(a["test"]), 1000)
        self.assertEqual(list(a["test"]), list(b["test"]))
        for k, expected in (("train", 30), ("val", 10), ("test", 10)):
            self.assertEqual(int(y.loc[a[k]].sum()), expected)


if __name__ == "__main__":
    unittest.main()
