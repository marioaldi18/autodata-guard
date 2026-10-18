"""Single source of truth for the dataset schema, data dictionary and validation rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

CATEGORIES: Tuple[str, ...] = ("Electronics", "Clothing", "Home", "Beauty", "Sports", "Books")
PAYMENT_METHODS: Tuple[str, ...] = ("card", "paypal", "bank_transfer", "cod")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # "string" | "int" | "float" | "date" | "category"
    description: str
    nullable: bool = False
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed: Optional[Tuple[str, ...]] = None
    unit: str = ""


SCHEMA: Tuple[ColumnSpec, ...] = (
    ColumnSpec("order_id", "string", "Unique order identifier", unique=True),
    ColumnSpec("customer_id", "string", "Customer identifier"),
    ColumnSpec("order_date", "date", "Order date (ISO YYYY-MM-DD); must not be after the reference date"),
    ColumnSpec("product_category", "category", "Product category", allowed=CATEGORIES),
    ColumnSpec("quantity", "int", "Units ordered", min_value=1, max_value=50, unit="units"),
    ColumnSpec("unit_price", "float", "Price per unit", min_value=0.5, max_value=5000, unit="currency"),
    ColumnSpec("discount_pct", "int", "Discount applied", min_value=0, max_value=90, unit="percent"),
    ColumnSpec("shipping_days", "int", "Days from order to delivery", nullable=True,
               min_value=0, max_value=30, unit="days"),
    ColumnSpec("payment_method", "category", "Payment method", allowed=PAYMENT_METHODS),
    ColumnSpec("customer_age", "int", "Customer age", nullable=True, min_value=18, max_value=100, unit="years"),
    ColumnSpec("total_amount", "float",
               "Order total; must equal quantity * unit_price * (1 - discount_pct/100) within 0.011",
               min_value=0, unit="currency"),
)

SCHEMA_BY_NAME: Dict[str, ColumnSpec] = {c.name: c for c in SCHEMA}
COLUMN_NAMES = [c.name for c in SCHEMA]
NUMERIC_DTYPES = ("int", "float")
