import unittest

import numpy as np
import pandas as pd

from autodataguard.config import resolve_config
from autodataguard.data.generator import generate_orders
from autodataguard.data.validation import coerce_types
from autodataguard.features.preprocessing import FEATURE_NAMES, FeaturePreprocessor


def typed_frame(seed=1, n=800):
    cfg = resolve_config({"data": {"n_rows": n}})["data"]
    raw, _ = generate_orders(cfg, seed)
    return coerce_types(raw)


class PreprocessingTests(unittest.TestCase):
    def test_output_has_no_nan_and_expected_columns(self):
        typed = typed_frame()
        out = FeaturePreprocessor().fit(typed).transform(typed)
        self.assertEqual(list(out.columns), list(FEATURE_NAMES))
        self.assertFalse(out.isna().any().any())
        self.assertTrue(np.isfinite(out.to_numpy()).all())

    def test_statistics_come_from_training_data_only(self):
        typed = typed_frame()
        train, test = typed.iloc[:600], typed.iloc[600:].copy()
        pre = FeaturePreprocessor().fit(train)
        medians_before = pre.medians_.copy()
        test["unit_price"] = 4000.0  # extreme test data must not change fitted state
        pre.transform(test)
        pd.testing.assert_series_equal(pre.medians_, medians_before)
        self.assertAlmostEqual(pre.medians_["unit_price"], train["unit_price"].where(train["unit_price"] >= 0.5).median())

    def test_unseen_category_and_invalid_values_are_handled(self):
        typed = typed_frame()
        pre = FeaturePreprocessor().fit(typed)
        odd = typed.iloc[:3].copy()
        odd.loc[odd.index[0], "product_category"] = "Gadgets"
        odd.loc[odd.index[1], "quantity"] = -4.0
        odd.loc[odd.index[2], "unit_price"] = np.nan
        out = pre.transform(odd)
        self.assertFalse(out.isna().any().any())
        self.assertEqual(float(out.iloc[2]["log_price_ratio"]) , float(pre.scaler_.transform(
            pre._features(pre._sanitize(odd)))[2][1]))

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            FeaturePreprocessor().transform(typed_frame(n=300))

    def test_all_missing_numeric_column_raises(self):
        typed = typed_frame(n=300)
        typed["customer_age"] = np.nan
        with self.assertRaises(ValueError):
            FeaturePreprocessor().fit(typed)


if __name__ == "__main__":
    unittest.main()
