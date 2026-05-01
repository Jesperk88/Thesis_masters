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
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from splits import get_session_split_indices

# Set random seed to guarantee identical splits to PyTorch models
SEED = 42


def load_embedding_map(pkl_path: str) -> dict:
    print(f"Loading embeddings map from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def sanitize_features(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def require_file(path: str, hint: str | None = None):
    if Path(path).exists():
        return

    message = f"Required file not found: {path}"
    if hint:
        message += f"\n{hint}"
    raise FileNotFoundError(message)


def assert_embedding_coverage(parquet_path: str, queries, note_idxs, q_map, i_map):
    missing_queries = list(dict.fromkeys(q for q in queries if q not in q_map))
    missing_items = list(dict.fromkeys(idx for idx in note_idxs if idx not in i_map))

    if not missing_queries and not missing_items:
        return

    examples = []
    if missing_queries:
        preview = ", ".join(repr(q) for q in missing_queries[:3])
        examples.append(f"{len(missing_queries)} queries, e.g. {preview}")
    if missing_items:
        preview = ", ".join(repr(idx) for idx in missing_items[:3])
        examples.append(f"{len(missing_items)} items, e.g. {preview}")

    raise ValueError(
        f"Embedding map does not cover {parquet_path}: {'; '.join(examples)}. "
        "Run src/prepare_search_test.py after the original search_train/DQA "
        "embedding generation so search_test embeddings are appended without "
        "regenerating the existing maps."
    )


def get_semantic_lr_data(parquet_path: str, emb_map: dict):
    require_file(parquet_path)
    print(f"Preparing semantic features for {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    
    queries = df["query"].to_list()
    note_idxs = df["note_idx"].to_list()
    
    q_map = emb_map["query"]
    i_map = emb_map["item"]
    ZERO_VEC = np.zeros(768, dtype=np.float32)

    assert_embedding_coverage(parquet_path, queries, note_idxs, q_map, i_map)

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
    sessions = df["session_idx"].to_numpy()

    X = np.column_stack((cos_sims, positions, is_left))
    return X, labels, sessions

def train_and_save_lr(X, y, parquet_path, env_name):
    # DQA still uses the internal split because there is no separate DQA test file.
    train_indices, val_indices = get_session_split_indices(parquet_path)

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    X_train = sanitize_features(X_train)
    X_val = sanitize_features(X_val)

    print(
        f"\nTraining Logistic Regression ({env_name}) on {len(y_train)} rows, "
        f"validating on {len(y_val)} rows..."
    )
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

def train_and_save_lr_train_test(train_path, test_path, emb_map, env_name):
    require_file(
        test_path,
        "Run python src/prepare_search_test.py to create search_test parquet "
        "and append only missing search_test embeddings.",
    )

    X_train, y_train, _ = get_semantic_lr_data(train_path, emb_map)
    X_test, y_test, _ = get_semantic_lr_data(test_path, emb_map)

    X_train = sanitize_features(X_train)
    X_test = sanitize_features(X_test)

    print(
        f"\nTraining Logistic Regression ({env_name}) on all "
        f"{len(y_train)} search_train rows..."
    )
    model = LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear")
    model.fit(X_train, y_train)

    test_preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_preds)
    loss = log_loss(y_test, test_preds)

    print(
        f"[{env_name}] Semantic Baseline - search_test AUC: {auc:.4f} | "
        f"search_test LogLoss: {loss:.4f}"
    )

    model_name = f"lr_{env_name.lower()}.pkl"
    with open(model_name, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved {model_name}")

if __name__ == "__main__":
    emb_map = load_embedding_map("data/embeddings_map.pkl")
    
    # Standard Environment
    std_train_path = "data/train_merged_text.parquet"
    std_test_path = "data/test_merged_text.parquet"
    train_and_save_lr_train_test(std_train_path, std_test_path, emb_map, "Standard")
    
    # DQA Environment
    dqa_path = "data/dqa_merged_text.parquet"
    X_dqa, y_dqa, _ = get_semantic_lr_data(dqa_path, emb_map)
    train_and_save_lr(X_dqa, y_dqa, dqa_path, "DQA")
    print("\nPhase 5 Complete. Semantic Baselines created.")
