"""Leakage-safe feature preprocessing: every statistic is learned from the training split only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ..data.schema import CATEGORIES, SCHEMA_BY_NAME

RAW_NUMERIC = ("quantity", "unit_price", "discount_pct", "shipping_days", "customer_age")
FEATURE_NAMES = ("log_quantity", "log_price_ratio", "discount_pct", "shipping_days", "customer_age", "log_revenue")


class FeaturePreprocessor:
    """Sanitise -> impute -> engineer -> robust-scale.

    Out-of-range values and invalid categories are treated as missing (they are reported by the
    validation engine, not modelled). ``log_price_ratio`` is the log of unit price relative to the
    training median price of the product category.
    """

    def _sanitize(self, typed: pd.DataFrame) -> pd.DataFrame:
        df = typed[["product_category", *RAW_NUMERIC]].copy()
        for col in RAW_NUMERIC:
            spec = SCHEMA_BY_NAME[col]
            v = df[col].astype(float)
            if spec.min_value is not None:
                v = v.mask(v < spec.min_value)
            if spec.max_value is not None:
                v = v.mask(v > spec.max_value)
            df[col] = v
        df["product_category"] = df["product_category"].where(df["product_category"].isin(CATEGORIES))
        return df

    def _features(self, clean: pd.DataFrame) -> pd.DataFrame:
        imp = clean[list(RAW_NUMERIC)].fillna(self.medians_)
        cat_median = clean["product_category"].map(self.category_median_price_).fillna(self.global_price_median_)
        ratio = np.log(imp["unit_price"] / cat_median).where(clean["unit_price"].notna(), 0.0)
        revenue = imp["quantity"] * imp["unit_price"] * (1 - imp["discount_pct"] / 100)
        return pd.DataFrame({
            "log_quantity": np.log1p(imp["quantity"]),
            "log_price_ratio": ratio,
            "discount_pct": imp["discount_pct"],
            "shipping_days": imp["shipping_days"],
            "customer_age": imp["customer_age"],
            "log_revenue": np.log1p(revenue),
        }, index=clean.index)[list(FEATURE_NAMES)]

    def fit(self, typed_train: pd.DataFrame) -> "FeaturePreprocessor":
        clean = self._sanitize(typed_train)
        self.medians_ = clean[list(RAW_NUMERIC)].median()
        if self.medians_.isna().any():
            raise ValueError("Cannot fit: a numeric column has no valid values in the training data")
        self.global_price_median_ = float(self.medians_["unit_price"])
        self.category_median_price_ = clean.groupby("product_category")["unit_price"].median().dropna().to_dict()
        self.scaler_ = RobustScaler().fit(self._features(clean))
        return self

    def transform(self, typed: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("FeaturePreprocessor must be fitted before transform")
        feats = self._features(self._sanitize(typed))
        return pd.DataFrame(self.scaler_.transform(feats), index=feats.index, columns=feats.columns)
