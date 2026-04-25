"""
run_dqa_ablation_studies.py
---------------------------
Runs DQA-side ablation studies for the thesis PAL model.

Ablations:
1. Full DQA-Aware PAL:
   - Relevance tower: full query + item embeddings
   - Examination tower: position + column + DQA embedding

2. DQA data, PAL without DQA embedding:
   - Relevance tower: full query + item embeddings
   - Examination tower: position + column only

3. Cosine-Relevance DQA PAL:
   - Relevance pathway: learned sigmoid over cosine(query_emb, item_emb)
   - Examination tower: position + column + projected DQA embedding

All splits are session-level using session_idx via GroupShuffleSplit.
"""

import os
import pickle
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss

from dataset import load_embedding_map
from models_final import DualTowerPAL


# ----------------------------
# Config
# ----------------------------

SEEDS = [42, 43, 44, 45, 46]

DQA_PARQUET = "data/dqa_merged_text.parquet"
EMBEDDING_MAP_PATH = "data/embeddings_map.pkl"

VAL_SPLIT = 0.10
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 5

LR_FULL_PAL = 0.005
LR_COSINE_PAL = 0.003
WEIGHT_DECAY = 1e-4

EMBEDDING_DIM = 768
ZERO_VEC = np.zeros(EMBEDDING_DIM, dtype=np.float32)

RESULTS_CSV = "dqa_ablation_results.csv"


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Session-level split
# ----------------------------

def get_session_split_indices(parquet_path: str, seed: int, val_size: float = VAL_SPLIT):
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


# ----------------------------
# Dataset
# ----------------------------

class DQAAblationDataset(Dataset):
    """
    DQA impression dataset with optional row indices.

    Returns:
      query_emb
      item_emb
      dqa_emb
      position
      is_left_col
      label
    """

    def __init__(self, parquet_path: str, embedding_map: dict, indices=None):
        self.df = pl.read_parquet(parquet_path)

        if indices is not None:
            self.df = self.df[indices]

        self.query_map = embedding_map["query"]
        self.item_map = embedding_map["item"]
        self.dqa_map = embedding_map["dqa"]

        self.queries = self.df["query"].to_list()
        self.note_idxs = self.df["note_idx"].to_list()
        self.dqa_texts = self.df["dqa_output"].to_list()
        self.positions = self.df["position"].to_list()
        self.left_cols = self.df["is_left_column"].cast(pl.Float32).to_list()
        self.labels = self.df["click"].cast(pl.Float32).to_list()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        query_str = self.queries[idx]
        note_idx = self.note_idxs[idx]
        dqa_str = self.dqa_texts[idx]

        query_emb = self.query_map.get(query_str, ZERO_VEC).astype(np.float32)
        item_emb = self.item_map.get(note_idx, ZERO_VEC).astype(np.float32)
        dqa_emb = self.dqa_map.get(dqa_str, ZERO_VEC).astype(np.float32)

        return {
            "query_emb": torch.tensor(query_emb, dtype=torch.float32),
            "item_emb": torch.tensor(item_emb, dtype=torch.float32),
            "dqa_emb": torch.tensor(dqa_emb, dtype=torch.float32),
            "position": torch.tensor([self.positions[idx]], dtype=torch.long),
            "is_left_col": torch.tensor([self.left_cols[idx]], dtype=torch.float32),
            "label": torch.tensor([self.labels[idx]], dtype=torch.float32),
        }


# ----------------------------
# Cosine-Relevance PAL model
# ----------------------------

class CosineRelevanceDQAPAL(nn.Module):
    """
    Lower-capacity DQA PAL variant.

    Relevance:
        P(R) = sigmoid(a * cosine(query, item) + b)

    Examination:
        P(E) = MLP(position_embedding, column, projected_dqa_embedding)

    Prediction:
        P(C) = P(E) * P(R)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_position: int = 60,
        position_emb_dim: int = 4,
        dqa_projection_dim: int = 32,
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=max_position + 1,
            embedding_dim=position_emb_dim
        )

        self.dqa_projection = nn.Sequential(
            nn.Linear(embedding_dim, dqa_projection_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        exam_input_dim = position_emb_dim + 1 + dqa_projection_dim

        self.examination_tower = nn.Sequential(
            nn.Linear(exam_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.cosine_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.cosine_bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, batch: dict) -> torch.Tensor:
        q = batch["query_emb"]
        i = batch["item_emb"]

        cosine = F.cosine_similarity(q, i, dim=1, eps=1e-8).unsqueeze(1)
        p_relevance = torch.sigmoid(self.cosine_scale * cosine + self.cosine_bias)

        pos_emb = self.position_embedding(batch["position"].squeeze(-1))
        col = batch["is_left_col"]
        dqa_proj = self.dqa_projection(batch["dqa_emb"])

        exam_input = torch.cat([pos_emb, col, dqa_proj], dim=-1)
        p_examination = self.examination_tower(exam_input)

        return p_examination * p_relevance


# ----------------------------
# Training and evaluation
# ----------------------------

def evaluate_model(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"]

            preds = model(batch)

            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

    auc = roc_auc_score(all_labels, all_preds)
    loss = log_loss(all_labels, all_preds)

    return auc, loss


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    checkpoint_path: str,
):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_val_auc = None
    epochs_no_improve = 0

    best_state_dict = None

    for epoch in range(max_epochs):
        model.train()
        train_loss_total = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"]

            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * labels.size(0)

        train_loss = train_loss_total / len(train_loader.dataset)
        val_auc, val_loss = evaluate_model(model, val_loader, device)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val LogLoss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            epochs_no_improve = 0

            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Improved. Saved {checkpoint_path}")

        else:
            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return best_val_auc, best_val_loss


def build_loaders(parquet_path, emb_map, seed):
    train_idx, val_idx = get_session_split_indices(parquet_path, seed=seed)

    train_dataset = DQAAblationDataset(parquet_path, emb_map, indices=train_idx)
    val_dataset = DQAAblationDataset(parquet_path, emb_map, indices=val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


# ----------------------------
# Main ablation runner
# ----------------------------

def run_single_seed(seed: int, emb_map: dict, device):
    print("\n" + "=" * 80)
    print(f"Running DQA ablations for seed {seed}")
    print("=" * 80)

    set_seed(seed)

    train_loader, val_loader, train_size, val_size = build_loaders(
        DQA_PARQUET,
        emb_map,
        seed
    )

    print(f"Train impressions: {train_size}")
    print(f"Validation impressions: {val_size}")

    results = []

    # Ablation 1: Full DQA PAL
    print("\n--- Ablation 1: Full DQA-Aware PAL ---")
    model_full = DualTowerPAL(use_dqa=True).to(device)

    auc, loss = train_model(
        model=model_full,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LR_FULL_PAL,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        checkpoint_path=f"ablation_full_dqa_pal_seed_{seed}.pth"
    )

    results.append({
        "seed": seed,
        "model": "Full DQA-Aware PAL",
        "roc_auc": auc,
        "logloss": loss
    })

    # Ablation 2: DQA data, but no DQA embedding in examination tower
    print("\n--- Ablation 2: DQA PAL without DQA embedding ---")
    model_no_dqa_emb = DualTowerPAL(use_dqa=False).to(device)

    auc, loss = train_model(
        model=model_no_dqa_emb,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LR_FULL_PAL,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        checkpoint_path=f"ablation_no_dqa_embedding_pal_seed_{seed}.pth"
    )

    results.append({
        "seed": seed,
        "model": "DQA Data PAL Without DQA Embedding",
        "roc_auc": auc,
        "logloss": loss
    })

    # Ablation 3: Lower-capacity cosine relevance PAL
    print("\n--- Ablation 3: Cosine-Relevance DQA PAL ---")
    model_cosine = CosineRelevanceDQAPAL().to(device)

    auc, loss = train_model(
        model=model_cosine,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LR_COSINE_PAL,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        checkpoint_path=f"ablation_cosine_relevance_dqa_pal_seed_{seed}.pth"
    )

    results.append({
        "seed": seed,
        "model": "Cosine-Relevance DQA PAL",
        "roc_auc": auc,
        "logloss": loss
    })

    return results


def print_summary(all_results):
    print("\n\n" + "=" * 80)
    print("DQA ABLATION SUMMARY")
    print("=" * 80)

    df = pl.DataFrame(all_results)

    summary = (
        df.group_by("model")
        .agg([
            pl.col("roc_auc").mean().alias("mean_auc"),
            pl.col("roc_auc").std().alias("std_auc"),
            pl.col("logloss").mean().alias("mean_logloss"),
            pl.col("logloss").std().alias("std_logloss"),
        ])
        .sort("mean_auc", descending=True)
    )

    print(summary)

    df.write_csv(RESULTS_CSV)
    print(f"\nSaved per-seed results to {RESULTS_CSV}")


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    emb_map = load_embedding_map(EMBEDDING_MAP_PATH)

    all_results = []

    for seed in SEEDS:
        seed_results = run_single_seed(seed, emb_map, device)
        all_results.extend(seed_results)

    print_summary(all_results)


if __name__ == "__main__":
    main()