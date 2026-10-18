import unittest

import numpy as np
import pandas as pd

from autodataguard.data.validation import validate

GOOD = {
    "order_id": "ORD-1", "customer_id": "C1", "order_date": "2025-01-10", "product_category": "Books",
    "quantity": "2", "unit_price": "10.00", "discount_pct": "10", "shipping_days": "3",
    "payment_method": "card", "customer_age": "30", "total_amount": "18.00",
}


def frame(*overrides):
    rows = []
    for i, o in enumerate(overrides):
        row = {**GOOD, "order_id": f"ORD-{i + 1}", **o}
        rows.append(row)
    return pd.DataFrame(rows).astype(object)


def rules(df):
    return set(validate(df).violations["rule"])


class ValidationTests(unittest.TestCase):
    def test_clean_row_has_no_violations(self):
        res = validate(frame({}))
        self.assertEqual(len(res.violations), 0)
        self.assertEqual(res.overall_score, 1.0)

    def test_missing_required_and_optional(self):
        self.assertEqual(rules(frame({"customer_id": None})), {"missing_required"})
        self.assertEqual(rules(frame({"customer_age": None})), {"missing_optional"})
        self.assertEqual(rules(frame({"customer_id": "   "})), {"missing_required"})

    def test_type_mismatch(self):
        self.assertIn("type_mismatch", rules(frame({"unit_price": "abc"})))
        self.assertIn("type_mismatch", rules(frame({"quantity": "2.5"})))  # non-integer in int column
        self.assertIn("type_mismatch", rules(frame({"order_date": "31/12/2025"})))
        self.assertIn("type_mismatch", rules(frame({"unit_price": "inf"})))

    def test_out_of_range_boundaries(self):
        self.assertEqual(rules(frame({"customer_age": "18", "total_amount": "18.00"})), set())  # boundary ok
        self.assertIn("out_of_range", rules(frame({"customer_age": "17"})))
        self.assertIn("out_of_range", rules(frame({"discount_pct": "91"})))
        self.assertIn("out_of_range", rules(frame({"quantity": "0"})))

    def test_invalid_category_is_exact_match(self):
        self.assertIn("invalid_category", rules(frame({"product_category": "books"})))
        self.assertIn("invalid_category", rules(frame({"payment_method": "crypto"})))

    def test_future_date(self):
        self.assertIn("future_date", rules(frame({"order_date": "2026-01-01"})))
        self.assertNotIn("future_date", rules(frame({"order_date": "2025-12-31"})))

    def test_duplicate_flags_later_occurrence_only(self):
        df = frame({}, {})
        df.loc[1, "order_id"] = "ORD-1"
        res = validate(df)
        self.assertEqual(res.rows_for_rule("duplicate_key"), [1])

    def test_total_mismatch_and_tolerance(self):
        self.assertIn("total_mismatch", rules(frame({"total_amount": "25.00"})))
        self.assertNotIn("total_mismatch", rules(frame({"total_amount": "18.01"})))
        # unparseable part: reported as type error, not as a consistency failure
        self.assertNotIn("total_mismatch", rules(frame({"unit_price": "abc"})))

    def test_empty_frame(self):
        res = validate(frame({}).iloc[0:0])
        self.assertEqual(len(res.violations), 0)
        self.assertEqual(res.dimension_scores["completeness"], 1.0)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            validate(frame({}).drop(columns=["quantity"]))

    def test_all_null_column_scores(self):
        df = frame({}, {}, {}, {})
        df["customer_age"] = np.nan
        res = validate(df)
        self.assertEqual(res.rows_for_rule("missing_optional"), [0, 1, 2, 3])
        self.assertAlmostEqual(res.dimension_scores["completeness"], 1 - 4 / 44)


if __name__ == "__main__":
    unittest.main()
