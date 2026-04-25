"""
splits.py
---------
Session-level train/validation split utilities.
Ensures all impressions from the same search session stay in the same split.
"""

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
VAL_SPLIT = 0.1


def get_session_split_indices(parquet_path: str, val_size: float = VAL_SPLIT, seed: int = SEED):
    """
    Returns train_idx, val_idx for a session-level split.
    Groups by session_idx so impressions from the same session cannot appear
    in both training and validation.
    """
    df = pl.read_parquet(parquet_path)

    if "session_idx" not in df.columns:
        raise ValueError(f"session_idx not found in {parquet_path}")

    groups = df["session_idx"].to_numpy()
    row_indices = np.arange(len(df))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed
    )

    train_idx, val_idx = next(splitter.split(row_indices, groups=groups))

    return train_idx, val_idx
