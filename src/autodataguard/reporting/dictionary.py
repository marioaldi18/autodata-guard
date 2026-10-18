"""Render the data dictionary (with validation rules) from the schema."""
from __future__ import annotations

from typing import Sequence

from ..data.schema import SCHEMA, ColumnSpec


def _rules(c: ColumnSpec) -> str:
    parts = ["required" if not c.nullable else "optional"]
    if c.unique:
        parts.append("unique")
    if c.min_value is not None:
        parts.append(f">= {c.min_value:g}")
    if c.max_value is not None:
        parts.append(f"<= {c.max_value:g}")
    if c.allowed:
        parts.append("one of: " + ", ".join(c.allowed))
    return "; ".join(parts)


def render_data_dictionary(schema: Sequence[ColumnSpec] = SCHEMA) -> str:
    lines = ["# Data Dictionary", "",
             "Generated from `src/autodataguard/data/schema.py`. Synthetic data only.", "",
             "| Column | Type | Unit | Validation rules | Description |", "|---|---|---|---|---|"]
    for c in schema:
        lines.append(f"| `{c.name}` | {c.dtype} | {c.unit or '-'} | {_rules(c)} | {c.description} |")
    lines += ["", "Cross-column rule: `total_amount` must equal `quantity * unit_price * (1 - discount_pct/100)` "
              "within 0.011.", ""]
    return "\n".join(lines)
