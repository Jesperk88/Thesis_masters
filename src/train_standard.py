"""
train_standard.py
-----------------
Trains the Standard Dual-Tower PAL model (use_dqa=False).
Environment: Standard search sessions.
The final Standard model is fit on all of search_train; search_test is reserved
for final evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules
from dataset import StandardImpressionDataset, load_embedding_map
from models_final import DualTowerPAL

# --- Hyperparameters ---
SEED = 42
BATCH_SIZE = 4096
LEARNING_RATE = 0.01
EPOCHS = 10
TRAIN_PATH = "data/train_merged_text.parquet"
MODEL_PATH = "standard_pal_best.pth"

def train_standard_model():
    torch.manual_seed(SEED)

    # 1. Setup Device for Apple M4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    embedding_map = load_embedding_map("data/embeddings_map.pkl")
    

    # 3. Train on the complete search_train split. Do not tune on search_test.
    train_dataset = StandardImpressionDataset(TRAIN_PATH, embedding_map)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    print(f"Training Standard Model on all {len(train_dataset)} search_train rows.")

    # 4. Initialize Model, Loss, Optimizer
    model = DualTowerPAL(use_dqa=False).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Move to M4 GPU
            query_emb = batch["query_emb"].to(device)
            item_emb = batch["item_emb"].to(device)
            position = batch["position"].to(device)
            is_left_col = batch["is_left_col"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model({
                "query_emb": query_emb,
                "item_emb": item_emb,
                "position": position,
                "is_left_col": is_left_col
            })

            # Calculate loss and backward pass
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Standard Model Training Complete. Saved {MODEL_PATH}")

if __name__ == "__main__":
    train_standard_model()
