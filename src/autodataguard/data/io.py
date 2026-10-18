"""CSV ingestion. Everything is read as text so validation sees the raw values."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV with all columns as strings; empty cells become missing values."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
