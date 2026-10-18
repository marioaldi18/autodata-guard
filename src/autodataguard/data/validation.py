"""Rule-based data validation driven by the schema.

Rules: missing_required / missing_optional (completeness), type_mismatch / out_of_range /
invalid_category / future_date (validity), duplicate_key (uniqueness), total_mismatch (consistency).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import NUMERIC_DTYPES, SCHEMA, ColumnSpec

VIOLATION_COLUMNS = ["row_index", "rule", "column", "dimension", "severity", "detail"]


@dataclass
class ValidationResult:
    violations: pd.DataFrame
    dimension_scores: Dict[str, float]
    n_rows: int
    n_columns: int

    @property
    def overall_score(self) -> float:
        return float(np.mean(list(self.dimension_scores.values())))

    def flagged_rows(self, severity: Optional[str] = None) -> list:
        v = self.violations
        if severity is not None:
            v = v[v["severity"] == severity]
        return sorted(v["row_index"].unique().tolist())

    def rows_for_rule(self, rule: str) -> list:
        v = self.violations
        return sorted(v.loc[v["rule"] == rule, "row_index"].unique().tolist())

    def counts_by_rule(self) -> pd.DataFrame:
        v = self.violations
        if v.empty:
            return pd.DataFrame(columns=["rule", "dimension", "severity", "violations", "rows_affected"])
        out = (v.groupby(["rule", "dimension", "severity"])
                 .agg(violations=("row_index", "size"), rows_affected=("row_index", "nunique"))
                 .reset_index()
                 .sort_values(["violations", "rule"], ascending=[False, True]))
        return out.reset_index(drop=True)


def _is_missing(s: pd.Series) -> pd.Series:
    blank = s.astype(object).where(s.notna(), "").astype(str).str.strip().eq("")
    return s.isna() | blank


def coerce_types(raw: pd.DataFrame, schema: Sequence[ColumnSpec] = SCHEMA) -> pd.DataFrame:
    """Parse columns to their schema types. Unparseable or non-finite values become NaN/NaT."""
    typed = pd.DataFrame(index=raw.index)
    for c in schema:
        s = raw[c.name]
        if c.dtype in NUMERIC_DTYPES:
            parsed = pd.to_numeric(s, errors="coerce").astype(float)
            typed[c.name] = parsed.where(np.isfinite(parsed))
        elif c.dtype == "date":
            typed[c.name] = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        else:
            typed[c.name] = s.astype(object)
    return typed


def validate(raw: pd.DataFrame, schema: Sequence[ColumnSpec] = SCHEMA,
             reference_date: str = "2025-12-31", total_tolerance: float = 0.011) -> ValidationResult:
    absent = [c.name for c in schema if c.name not in raw.columns]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")

    n = len(raw)
    ref = pd.Timestamp(reference_date)
    typed = coerce_types(raw, schema)
    records: List[pd.DataFrame] = []

    def add(rule: str, column: str, mask: pd.Series, severity: str, dimension: str, detail: str) -> None:
        rows = raw.index[np.asarray(mask, dtype=bool)]
        if len(rows) == 0:
            return
        records.append(pd.DataFrame({"row_index": rows, "rule": rule, "column": column,
                                     "dimension": dimension, "severity": severity, "detail": detail}))

    for c in schema:
        col = raw[c.name]
        missing = _is_missing(col)
        if c.nullable:
            add("missing_optional", c.name, missing, "warning", "completeness", "optional value missing")
        else:
            add("missing_required", c.name, missing, "error", "completeness", "required value missing")
        present = ~missing

        if c.dtype in NUMERIC_DTYPES:
            parsed = typed[c.name]
            bad_type = present & parsed.isna()
            if c.dtype == "int":
                bad_type = bad_type | (present & parsed.notna() & (parsed % 1 != 0))
            add("type_mismatch", c.name, bad_type, "error", "validity", f"not a valid {c.dtype}")
            ok = parsed.notna() & ~bad_type
            if c.min_value is not None:
                add("out_of_range", c.name, ok & (parsed < c.min_value), "error", "validity",
                    f"below minimum {c.min_value}")
            if c.max_value is not None:
                add("out_of_range", c.name, ok & (parsed > c.max_value), "error", "validity",
                    f"above maximum {c.max_value}")
        elif c.dtype == "date":
            parsed = typed[c.name]
            add("type_mismatch", c.name, present & parsed.isna(), "error", "validity", "not a valid YYYY-MM-DD date")
            add("future_date", c.name, parsed.notna() & (parsed > ref), "error", "validity",
                f"after reference date {reference_date}")
        elif c.dtype == "category" and c.allowed is not None:
            bad = present & ~col.astype(object).isin(c.allowed)
            add("invalid_category", c.name, bad, "error", "validity", "value not in allowed set")

        if c.unique:
            add("duplicate_key", c.name, present & col.duplicated(keep="first"), "error", "uniqueness",
                "repeat of an earlier key")

    parts = typed[["quantity", "unit_price", "discount_pct", "total_amount"]]
    complete = parts.notna().all(axis=1)
    expected = typed["quantity"] * typed["unit_price"] * (1 - typed["discount_pct"] / 100)
    add("total_mismatch", "total_amount", complete & ((typed["total_amount"] - expected).abs() > total_tolerance),
        "error", "consistency", "total_amount != quantity * unit_price * (1 - discount_pct/100)")

    violations = (pd.concat(records, ignore_index=True) if records
                  else pd.DataFrame(columns=VIOLATION_COLUMNS))
    scores = _dimension_scores(violations, n, len(schema))
    return ValidationResult(violations, scores, n, len(schema))


def _dimension_scores(v: pd.DataFrame, n_rows: int, n_cols: int) -> Dict[str, float]:
    if n_rows == 0:
        return {"completeness": 1.0, "validity": 1.0, "uniqueness": 1.0, "consistency": 1.0}
    cells = n_rows * n_cols

    def cell_count(dim: str) -> int:
        sub = v[v["dimension"] == dim]
        return int(sub[["row_index", "column"]].drop_duplicates().shape[0])

    def row_count(rule: str) -> int:
        return int(v.loc[v["rule"] == rule, "row_index"].nunique())

    return {
        "completeness": 1.0 - cell_count("completeness") / cells,
        "validity": 1.0 - cell_count("validity") / cells,
        "uniqueness": 1.0 - row_count("duplicate_key") / n_rows,
        "consistency": 1.0 - row_count("total_mismatch") / n_rows,
    }
