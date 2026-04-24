"""
dataset.py
----------
Custom PyTorch Dataset classes for the Dual-Tower PAL model.
Reads flattened impression Parquet files and looks up pre-computed
768-dimensional embeddings from the pkl map.

Two classes:
  - StandardImpressionDataset : standard sessions (no DQA)
  - DQAImpressionDataset      : DQA-present sessions (includes dqa_output embedding)
"""

import pickle
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

EMBEDDING_DIM = 768
ZERO_VEC = np.zeros(EMBEDDING_DIM, dtype=np.float32)

def load_embedding_map(pkl_path: str) -> dict:
    """Load the pre-computed embedding maps from disk."""
    print(f"Loading embeddings map from {pkl_path} into memory...")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

class StandardImpressionDataset(Dataset):
    """
    Dataset for standard (non-DQA) search sessions.

    Returns per impression:
      - query_emb   : (768,) float tensor
      - item_emb    : (768,) float tensor
      - position    : (1,)   long tensor   [1-indexed rank]
      - is_left_col : (1,)   float tensor  [1.0 = left, 0.0 = right]
      - label       : (1,)   float tensor  [click = 1, no click = 0]
    """
    def __init__(self, parquet_path: str, embedding_map: dict):
        print(f"Loading Standard dataset from {parquet_path}...")
        self.df = pl.read_parquet(parquet_path)
        self.query_map = embedding_map["query"]
        self.item_map  = embedding_map["item"]

        # Pre-extract columns as lists for O(1) indexing during __getitem__
        self.queries    = self.df["query"].to_list()
        self.note_idxs  = self.df["note_idx"].to_list()
        self.positions  = self.df["position"].to_list()
        self.left_cols  = self.df["is_left_column"].cast(pl.Float32).to_list()
        self.labels     = self.df["click"].cast(pl.Float32).to_list()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        query_str = self.queries[idx]
        note_idx  = self.note_idxs[idx]

        # Exact dictionary lookup with zero-vector fallback
        query_emb = self.query_map.get(query_str, ZERO_VEC).astype(np.float32)
        item_emb = self.item_map.get(note_idx, ZERO_VEC).astype(np.float32)

        return {
            "query_emb":   torch.tensor(query_emb),
            "item_emb":    torch.tensor(item_emb),
            "position":    torch.tensor([self.positions[idx]], dtype=torch.long),
            "is_left_col": torch.tensor([self.left_cols[idx]], dtype=torch.float32),
            "label":       torch.tensor([self.labels[idx]], dtype=torch.float32),
        }


class DQAImpressionDataset(Dataset):
    """
    Dataset for DQA-present search sessions.

    Identical to StandardImpressionDataset but additionally returns:
      - dqa_emb : (768,) float tensor — embedding of the DQA module output text
    """
    def __init__(self, parquet_path: str, embedding_map: dict):
        print(f"Loading DQA dataset from {parquet_path}...")
        self.df = pl.read_parquet(parquet_path)
        self.query_map = embedding_map["query"]
        self.item_map  = embedding_map["item"]
        self.dqa_map   = embedding_map["dqa"]

        # Pre-extract columns
        self.queries   = self.df["query"].to_list()
        self.note_idxs = self.df["note_idx"].to_list()
        self.dqa_texts = self.df["dqa_output"].to_list()
        self.positions = self.df["position"].to_list()
        self.left_cols = self.df["is_left_column"].cast(pl.Float32).to_list()
        self.labels    = self.df["click"].cast(pl.Float32).to_list()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        query_str = self.queries[idx]
        note_idx  = self.note_idxs[idx]
        dqa_str   = self.dqa_texts[idx]

        # Exact dictionary lookup with zero-vector fallback
        query_emb = self.query_map.get(query_str, ZERO_VEC).astype(np.float32)
        item_emb  = self.item_map.get(note_idx, ZERO_VEC).astype(np.float32)
        dqa_emb   = self.dqa_map.get(dqa_str, ZERO_VEC).astype(np.float32)

        return {
            "query_emb":   torch.tensor(query_emb),
            "item_emb":    torch.tensor(item_emb),
            "dqa_emb":     torch.tensor(dqa_emb),
            "position":    torch.tensor([self.positions[idx]], dtype=torch.long),
            "is_left_col": torch.tensor([self.left_cols[idx]], dtype=torch.float32),
            "label":       torch.tensor([self.labels[idx]], dtype=torch.float32),
        }