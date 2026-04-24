"""
evaluate.py
-----------
Final Metrics & Curve Extraction.
Evaluates the PAL Models and Semantic Logistic Regression baselines on the 
exact same validation batches to guarantee fair comparison.

Extracts the learned Position Bias curve from the PAL examination pathway
and outputs it as 'Fig_4_learned_bias_comparison.pdf'.
"""

import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss

# Import custom architecture
from dataset import StandardImpressionDataset, DQAImpressionDataset, load_embedding_map
from models_final import DualTowerPAL

VAL_SPLIT = 0.1
BATCH_SIZE = 2048

def get_val_loader(dataset):
    """Recreates the exact validation split used in training."""
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    return DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate_environment(env_name, model_pal, model_lr, loader, device, is_dqa=False):
    model_pal.eval()
    
    all_labels = []
    pal_preds = []
    lr_preds = []

    print(f"Evaluating {env_name} Models...")
    with torch.no_grad():
        for batch in loader:
            # --- PAL Inference ---
            q_emb = batch["query_emb"].to(device)
            i_emb = batch["item_emb"].to(device)
            pos = batch["position"].to(device)
            col = batch["is_left_col"].to(device)
            labels = batch["label"].to(device)
            
            inputs = {
                "query_emb": q_emb, "item_emb": i_emb, 
                "position": pos, "is_left_col": col
            }
            if is_dqa:
                inputs["dqa_emb"] = batch["dqa_emb"].to(device)

            pal_batch_preds = model_pal(inputs).cpu().numpy().flatten()
            
            # --- LR Inference ---
            # Compute cosine similarity efficiently on GPU/CPU natively
            cos_sim = F.cosine_similarity(q_emb, i_emb, dim=1).cpu().numpy()
            pos_np = pos.cpu().numpy().flatten()
            col_np = col.cpu().numpy().flatten()
            
            # Reconstruct the feature matrix for LR
            X_lr = np.column_stack((cos_sim, pos_np, col_np))
            lr_batch_preds = model_lr.predict_proba(X_lr)[:, 1]

            all_labels.extend(labels.cpu().numpy().flatten())
            pal_preds.extend(pal_batch_preds)
            lr_preds.extend(lr_batch_preds)

    # Calculate metrics
    pal_auc = roc_auc_score(all_labels, pal_preds)
    pal_loss = log_loss(all_labels, pal_preds)
    lr_auc = roc_auc_score(all_labels, lr_preds)
    lr_loss = log_loss(all_labels, lr_preds)

    return (lr_auc, lr_loss), (pal_auc, pal_loss)

def extract_and_plot_curves(pal_std, pal_dqa, emb_map, device):
    print("Extracting internal P(E) representations for Figure 4...")
    
    # 1. Dummy Layout Setup: Positions 1 through 10, Left Column
    pos_dummy = torch.arange(1, 11, dtype=torch.long).unsqueeze(1).to(device) # Shape (10, 1)
    col_dummy = torch.ones(10, 1, dtype=torch.float32).to(device)             # Shape (10, 1)
    
    # 2. Extract Standard P(E)
    pos_emb_std = pal_std.position_embedding(pos_dummy.squeeze(-1))
    exam_in_std = torch.cat([pos_emb_std, col_dummy], dim=-1)
    p_e_std = pal_std.examination_tower(exam_in_std).detach().cpu().numpy().flatten()

    # 3. Extract DQA P(E)
    # Get the average semantic space of a DQA response
    all_dqa_vecs = list(emb_map['dqa'].values())
    avg_dqa_vec = np.mean(all_dqa_vecs, axis=0)
    dqa_tensor = torch.tensor(avg_dqa_vec, dtype=torch.float32).unsqueeze(0).repeat(10, 1).to(device)

    pos_emb_dqa = pal_dqa.position_embedding(pos_dummy.squeeze(-1))
    exam_in_dqa = torch.cat([pos_emb_dqa, col_dummy, dqa_tensor], dim=-1)
    p_e_dqa = pal_dqa.examination_tower(exam_in_dqa).detach().cpu().numpy().flatten()

    # 4. Generate the Thesis Graph
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(1, 11), p_e_std, marker='o', markersize=8, 
             linestyle='-', linewidth=2, color='#008080', label='Learned: Standard Layout')
    plt.plot(range(1, 11), p_e_dqa, marker='s', markersize=8, 
             linestyle='--', linewidth=2, color='#D11141', label='Learned: DQA-Present')

    plt.title("Neural Network's Internal Representation of Examination Bias", fontsize=16, pad=20)
    plt.xlabel("Organic Rank Position", fontsize=12)
    plt.ylabel("Learned Absolute Examination Probability $P(E)$", fontsize=12)
    plt.xticks(range(1, 11))
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save exactly as required for the thesis LaTeX
    file_name = 'Fig_4_learned_bias_comparison.pdf'
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Saved learned bias curve to {file_name}")

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluation running on: {device}")

    # 1. Load Embeddings and Datasets
    emb_map = load_embedding_map("data/embeddings_map.pkl")
    dataset_std = StandardImpressionDataset("data/train_merged_text.parquet", emb_map)
    dataset_dqa = DQAImpressionDataset("data/dqa_merged_text.parquet", emb_map)

    loader_std = get_val_loader(dataset_std)
    loader_dqa = get_val_loader(dataset_dqa)

    # 2. Load the trained PAL models
    pal_std = DualTowerPAL(use_dqa=False).to(device)
    pal_std.load_state_dict(torch.load("standard_pal_best.pth", map_location=device))
    
    pal_dqa = DualTowerPAL(use_dqa=True).to(device)
    pal_dqa.load_state_dict(torch.load("dqa_pal_best.pth", map_location=device))

    # 3. Load the Logistic Regression models
    with open("lr_standard.pkl", "rb") as f:
        lr_std = pickle.load(f)
    with open("lr_dqa.pkl", "rb") as f:
        lr_dqa = pickle.load(f)

    # 4. Evaluate Environment 1: Standard
    std_lr_metrics, std_pal_metrics = evaluate_environment(
        "Standard", pal_std, lr_std, loader_std, device, is_dqa=False
    )

    # 5. Evaluate Environment 2: DQA-Present
    dqa_lr_metrics, dqa_pal_metrics = evaluate_environment(
        "DQA", pal_dqa, lr_dqa, loader_dqa, device, is_dqa=True
    )

    # 6. Extract the Plot
    extract_and_plot_curves(pal_std, pal_dqa, emb_map, device)

    # 7. Print the Thesis Results Table
    print("\n\n" + "=" * 70)
    print(f"{'Model Architecture':<32} | {'Environment':<12} | {'ROC-AUC':<8} | {'LogLoss':<8}")
    print("-" * 70)
    print(f"{'Logistic Regression (Semantic)':<32} | {'Standard':<12} | {std_lr_metrics[0]:.4f}   | {std_lr_metrics[1]:.4f}")
    print(f"{'Dual-Tower PAL Baseline':<32} | {'Standard':<12} | {std_pal_metrics[0]:.4f}   | {std_pal_metrics[1]:.4f}")
    print("-" * 70)
    print(f"{'Logistic Regression (Semantic)':<32} | {'DQA-Present':<12} | {dqa_lr_metrics[0]:.4f}   | {dqa_lr_metrics[1]:.4f}")
    print(f"{'DQA-Aware Dual-Tower PAL':<32} | {'DQA-Present':<12} | {dqa_pal_metrics[0]:.4f}   | {dqa_pal_metrics[1]:.4f}")
    print("=" * 70)