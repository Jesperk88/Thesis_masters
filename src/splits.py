"""
splits.py
---------
Random impression-level train/validation split utilities.

Creates a reproducible 90/10 split after the data has been flattened to the
impression level. This is used consistently for the logistic regression
baselines and PAL models.
"""

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

SEED = 42
VAL_SPLIT = 0.1


def get_session_split_indices(parquet_path: str, val_size: float = VAL_SPLIT, seed: int = SEED):
    """
    Returns train_idx, val_idx for a random impression-level split.

    The function name is kept the same so existing scripts do not need to be
    changed. Unlike the previous version, this does NOT group by session_idx.
    """
    df = pl.read_parquet(parquet_path)

    row_indices = np.arange(len(df))

    train_idx, val_idx = train_test_split(
        row_indices,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=None
    )

    return train_idx, val_idx