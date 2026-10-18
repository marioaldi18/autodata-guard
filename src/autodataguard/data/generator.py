"""Synthetic e-commerce order generator.

Produces a raw (all-string) table plus a separate ground-truth label table:

* statistical anomalies: valid-looking rows with unusual values (price spike, bulk quantity,
  extreme discount, slow shipping). Totals stay consistent, so rule checks do not catch them.
* data-quality issues: rule violations (missing, type, range, category, future date, total mismatch).
* duplicates: exact copies of clean rows appended at the end (simulating a re-ingested batch).

Labels are returned separately and are never part of the feature table.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .schema import CATEGORIES, PAYMENT_METHODS, SCHEMA

CATEGORY_PROBS = (0.20, 0.20, 0.20, 0.15, 0.15, 0.10)
CATEGORY_MEDIAN_PRICE = {
    "Electronics": 250.0, "Clothing": 45.0, "Home": 80.0,
    "Beauty": 25.0, "Sports": 60.0, "Books": 15.0,
}
PAYMENT_PROBS = (0.55, 0.25, 0.12, 0.08)
DISCOUNTS = (0, 0, 0, 5, 10, 15, 20, 25, 30)

ANOMALY_TYPES = ("price_spike", "bulk_quantity", "extreme_discount", "slow_shipping")
DQ_ISSUE_TYPES = (
    "missing_required", "missing_optional", "type_error", "out_of_range",
    "invalid_category", "future_date", "total_mismatch",
)
DUPLICATE_TYPE = "duplicate_key"

_NUMERIC_COLS = ["quantity", "unit_price", "discount_pct", "shipping_days", "customer_age"]


def _base_orders(rng: np.random.Generator, n: int, cfg: Dict[str, Any]) -> pd.DataFrame:
    start, ref = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["reference_date"])
    span = (ref - start).days + 1
    dates = start + pd.to_timedelta(rng.integers(0, span, n), unit="D")
    n_customers = max(10, n // 5)
    cats = rng.choice(CATEGORIES, size=n, p=CATEGORY_PROBS)
    median = np.array([CATEGORY_MEDIAN_PRICE[c] for c in cats])
    price = np.clip(np.round(median * np.exp(0.4 * rng.standard_normal(n)), 2), 0.5, 5000.0)
    return pd.DataFrame({
        "order_id": [f"ORD-{i + 1:06d}" for i in range(n)],
        "customer_id": [f"C{c:05d}" for c in rng.integers(1, n_customers + 1, n)],
        "order_date": dates,
        "product_category": cats,
        "quantity": np.clip(1 + rng.poisson(1.2, n), 1, 8).astype(int),
        "unit_price": price,
        "discount_pct": rng.choice(DISCOUNTS, n).astype(int),
        "shipping_days": np.clip(np.round(rng.normal(4, 1.5, n)), 1, 10).astype(int),
        "payment_method": rng.choice(PAYMENT_METHODS, size=n, p=PAYMENT_PROBS),
        "customer_age": np.clip(np.round(rng.normal(38, 11, n)), 18, 80).astype(int),
    })


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray,
                      anomaly_type: np.ndarray) -> None:
    types = rng.permutation(np.resize(np.array(ANOMALY_TYPES, dtype=object), len(idx)))
    for i, t in zip(idx, types):
        anomaly_type[i] = t
        if t == "price_spike":
            df.at[i, "unit_price"] = min(round(float(df.at[i, "unit_price"]) * rng.uniform(3.5, 12.0), 2), 4999.0)
        elif t == "bulk_quantity":
            df.at[i, "quantity"] = int(rng.integers(15, 41))
        elif t == "extreme_discount":
            df.at[i, "discount_pct"] = int(rng.integers(60, 86))
        elif t == "slow_shipping":
            df.at[i, "shipping_days"] = int(rng.integers(18, 31))


def _to_strings(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": df["order_id"],
        "customer_id": df["customer_id"],
        "order_date": df["order_date"].dt.strftime("%Y-%m-%d"),
        "product_category": df["product_category"],
        "quantity": df["quantity"].astype(int).astype(str),
        "unit_price": df["unit_price"].map("{:.2f}".format),
        "discount_pct": df["discount_pct"].astype(int).astype(str),
        "shipping_days": df["shipping_days"].astype(int).astype(str),
        "payment_method": df["payment_method"],
        "customer_age": df["customer_age"].astype(int).astype(str),
        "total_amount": df["total_amount"].map("{:.2f}".format),
    }).astype(object)


def _inject_dq_issues(raw: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray,
                      dq_type: np.ndarray, cfg: Dict[str, Any]) -> None:
    if len(idx) == 0:
        return
    ref = pd.Timestamp(cfg["reference_date"])
    required = [c.name for c in SCHEMA if not c.nullable and c.name != "order_id"]
    optional = [c.name for c in SCHEMA if c.nullable]
    types = rng.permutation(np.resize(np.array(DQ_ISSUE_TYPES, dtype=object), len(idx)))
    for i, t in zip(idx, types):
        dq_type[i] = t
        if t == "missing_required":
            raw.at[i, str(rng.choice(required))] = None
        elif t == "missing_optional":
            raw.at[i, str(rng.choice(optional))] = None
        elif t == "type_error":
            col = str(rng.choice(_NUMERIC_COLS + ["order_date"]))
            raw.at[i, col] = "not_a_date" if col == "order_date" else str(rng.choice(["N/A", "unknown", "abc", "?", "1,5"]))
        elif t == "out_of_range":
            col = str(rng.choice(_NUMERIC_COLS))
            raw.at[i, col] = {
                "quantity": lambda: str(int(rng.integers(-5, 1))),
                "unit_price": lambda: f"{-rng.uniform(1, 100):.2f}",
                "discount_pct": lambda: str(int(rng.integers(101, 151))),
                "shipping_days": lambda: str(int(rng.integers(-5, 0))),
                "customer_age": lambda: str(int(rng.choice([5, 12, 130, 150]))),
            }[col]()
        elif t == "invalid_category":
            raw.at[i, "product_category"] = str(rng.choice(["Electronic", "Gadgets", "UNKNOWN", "clothing "]))
        elif t == "future_date":
            raw.at[i, "order_date"] = (ref + pd.Timedelta(days=int(rng.integers(1, 366)))).strftime("%Y-%m-%d")
        elif t == "total_mismatch":
            raw.at[i, "total_amount"] = f"{float(raw.at[i, 'total_amount']) * rng.uniform(1.2, 3.0):.2f}"


def generate_orders(data_cfg: Dict[str, Any], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw, labels)``; ``labels`` is aligned to ``raw`` by ``row_index``."""
    n = int(data_cfg["n_rows"])
    n_anom = int(round(n * data_cfg["anomaly_rate"]))
    n_dq = int(round(n * data_cfg["dq_issue_rate"]))
    n_dup = int(round(n * data_cfg["duplicate_rate"]))
    if n_anom + n_dq + n_dup > n:
        raise ValueError("Injected rows exceed n_rows")

    rng = np.random.default_rng(seed)
    base = _base_orders(rng, n, data_cfg)
    order = rng.permutation(n)
    anom_idx = order[:n_anom]
    dq_idx = order[n_anom:n_anom + n_dq]
    dup_src = order[n_anom + n_dq:n_anom + n_dq + n_dup]  # clean rows only

    anomaly_type = np.full(n, "", dtype=object)
    _inject_anomalies(base, rng, anom_idx, anomaly_type)
    base["total_amount"] = np.round(base["quantity"] * base["unit_price"] * (1 - base["discount_pct"] / 100), 2)

    raw = _to_strings(base)
    dq_type = np.full(n, "", dtype=object)
    _inject_dq_issues(raw, rng, dq_idx, dq_type, data_cfg)

    dups = raw.iloc[dup_src].copy()
    raw = pd.concat([raw, dups], ignore_index=True)

    total = n + n_dup
    labels = pd.DataFrame({
        "row_index": np.arange(total),
        "is_anomaly": np.concatenate([anomaly_type != "", np.zeros(n_dup, dtype=bool)]),
        "anomaly_type": np.concatenate([anomaly_type, np.full(n_dup, "", dtype=object)]),
        "has_dq_issue": np.concatenate([dq_type != "", np.ones(n_dup, dtype=bool)]),
        "dq_issue_type": np.concatenate([dq_type, np.full(n_dup, DUPLICATE_TYPE, dtype=object)]),
    })
    return raw, labels
