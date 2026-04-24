"""
train_dqa.py
------------
Trains the DQA-Aware Dual-Tower PAL model (use_dqa=True).
Environment: DQA-present search sessions.
Hyperparameters: Batch Size 256, LR 0.005, Weight Decay 1e-4, Max Epochs 50.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np

# Import your custom modules
from dataset import DQAImpressionDataset, load_embedding_map
from models_final import DualTowerPAL

# --- Hyperparameters ---
BATCH_SIZE = 256
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 5
VAL_SPLIT = 0.1

def train_dqa_model():
    # 1. Setup Device for Apple M4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    embedding_map = load_embedding_map("data/embeddings_map.pkl")
    full_dataset = DQAImpressionDataset("data/dqa_merged_text.parquet", embedding_map)

    # 3. Create Train/Validation Split (90/10)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training DQA-Aware Model on {train_size} rows, validating on {val_size} rows.")

    # 4. Initialize Model, Loss, Optimizer
    model = DualTowerPAL(use_dqa=True).to(device)
    criterion = nn.BCELoss()
    # Adding weight_decay acts as L2 regularization to prevent overfitting on the smaller dataset
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 5. Training Loop with Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Move to M4 GPU
            query_emb = batch["query_emb"].to(device)
            item_emb = batch["item_emb"].to(device)
            dqa_emb = batch["dqa_emb"].to(device)
            position = batch["position"].to(device)
            is_left_col = batch["is_left_col"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward pass (DQA environment)
            preds = model({
                "query_emb": query_emb,
                "item_emb": item_emb,
                "dqa_emb": dqa_emb,
                "position": position,
                "is_left_col": is_left_col
            })

            # Calculate loss and backward pass
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_targets = []
        val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                query_emb = batch["query_emb"].to(device)
                item_emb = batch["item_emb"].to(device)
                dqa_emb = batch["dqa_emb"].to(device)
                position = batch["position"].to(device)
                is_left_col = batch["is_left_col"].to(device)
                labels = batch["label"].to(device)

                preds = model({
                    "query_emb": query_emb,
                    "item_emb": item_emb,
                    "dqa_emb": dqa_emb,
                    "position": position,
                    "is_left_col": is_left_col
                })

                val_targets.extend(labels.cpu().numpy().flatten())
                val_preds.extend(preds.cpu().numpy().flatten())

        # Calculate epoch metrics
        val_loss = log_loss(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val LogLoss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "dqa_pal_best.pth")
            print("  --> Model improved. Weights saved to dqa_pal_best.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("DQA-Aware Model Training Complete.")

if __name__ == "__main__":
    train_dqa_model()