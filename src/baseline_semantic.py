"""
baseline_semantic.py
--------------------
Updates the naive baseline to a Semantic-Aware Logistic Regression.
Extracts Cosine Similarity between Query and Item embeddings as a new feature,
combining it with Rank and Column Placement.

Saves two trained LR models for the final evaluation table.
"""

import numpy as np
import polars as pl
import pickle
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import os

# Set random seed to guarantee identical splits to PyTorch models
SEED = 42

def load_embedding_map(pkl_path: str) -> dict:
    print(f"Loading embeddings map from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def get_semantic_lr_data(parquet_path: str, emb_map: dict):
    print(f"Preparing semantic features for {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    
    queries = df["query"].to_list()
    note_idxs = df["note_idx"].to_list()
    
    q_map = emb_map["query"]
    i_map = emb_map["item"]
    ZERO_VEC = np.zeros(768, dtype=np.float32)

    # Fetch vectors
    q_embs = np.array([q_map.get(q, ZERO_VEC) for q in queries])
    i_embs = np.array([i_map.get(idx, ZERO_VEC) for idx in note_idxs])

    # Compute Cosine Similarity
    print("Computing cosine similarities...")
    norms_q = np.linalg.norm(q_embs, axis=1)
    norms_i = np.linalg.norm(i_embs, axis=1)
    
    # Avoid division by zero
    valid = (norms_q > 0) & (norms_i > 0)
    cos_sims = np.zeros(len(df), dtype=np.float32)
    cos_sims[valid] = np.sum(q_embs[valid] * i_embs[valid], axis=1) / (norms_q[valid] * norms_i[valid])

    # Build X matrix: [Cosine_Sim, Position, Is_Left_Col]
    positions = df["position"].to_numpy()
    is_left = df["is_left_column"].cast(pl.Float32).to_numpy()
    labels = df["click"].to_numpy()

    X = np.column_stack((cos_sims, positions, is_left))
    return X, labels

def train_and_save_lr(X, y, env_name):
    # Replicate the exact 90/10 split logic used in PyTorch random_split
    dataset_len = len(y)
    val_size = int(dataset_len * 0.1)
    train_size = dataset_len - val_size
    
    indices = torch.randperm(dataset_len, generator=torch.Generator().manual_seed(SEED)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # --- THE FIX ---
    # 1. Force float64 precision and wipe out any hidden infinite/NaN values
    X_train = np.nan_to_num(X_train.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nTraining Logistic Regression ({env_name})...")
    # 2. Switch solver from 'lbfgs' to 'liblinear' which is mathematically immune to matmul overflow
    model = LogisticRegression(max_iter=1000, random_state=SEED, solver='liblinear')
    model.fit(X_train, y_train)
    
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    loss = log_loss(y_val, val_preds)
    
    print(f"[{env_name}] Semantic Baseline - Val AUC: {auc:.4f} | Val LogLoss: {loss:.4f}")
    
    # Save the model
    model_name = f"lr_{env_name.lower()}.pkl"
    with open(model_name, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {model_name}")

if __name__ == "__main__":
    emb_map = load_embedding_map("data/embeddings_map.pkl")
    
    # Standard Environment
    X_std, y_std = get_semantic_lr_data("data/train_merged_text.parquet", emb_map)
    train_and_save_lr(X_std, y_std, "Standard")
    
    # DQA Environment
    X_dqa, y_dqa = get_semantic_lr_data("data/dqa_merged_text.parquet", emb_map)
    train_and_save_lr(X_dqa, y_dqa, "DQA")
    print("\nPhase 5 Complete. Semantic Baselines created.")