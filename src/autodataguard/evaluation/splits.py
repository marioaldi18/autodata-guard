"""Stratified train/validation/test split (labels are used only for splitting and evaluation)."""
from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def split_indices(y: pd.Series, val_size: float, test_size: float, seed: int) -> Dict[str, pd.Index]:
    idx = y.index.to_numpy()
    holdout = val_size + test_size
    train_idx, hold_idx = train_test_split(idx, test_size=holdout, stratify=y.to_numpy(), random_state=seed)
    hold_y = y.loc[hold_idx].to_numpy()
    val_idx, test_idx = train_test_split(hold_idx, test_size=test_size / holdout, stratify=hold_y,
                                         random_state=seed)
    return {"train": pd.Index(train_idx), "val": pd.Index(val_idx), "test": pd.Index(test_idx)}
