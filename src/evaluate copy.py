"""
evaluate.py
-----------
Final Metrics & Curve Extraction.
Evaluates the Standard models on held-out search_test and the DQA-present
models on the same internal DQA validation split used during training.

Extracts the learned Position Bias curve from the PAL examination pathway
and outputs it as 'Fig_4_learned_bias_true_layout.pdf'.
"""

import csv
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss

# Import custom architecture
from dataset import StandardImpressionDataset, DQAImpressionDataset, load_embedding_map
from models_final import DualTowerPAL
from splits import get_session_split_indices

VAL_SPLIT = 0.1
BATCH_SIZE = 2048
LEARNED_BIAS_CSV = "Fig_4_learned_bias_values.csv"
LEARNED_BIAS_TEX = "Fig_4_learned_bias_values.tex"

def get_test_loader_standard(parquet_path, emb_map):
    dataset = StandardImpressionDataset(parquet_path, emb_map)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def get_val_loader_dqa(parquet_path, emb_map):
    _, val_idx = get_session_split_indices(parquet_path)
    dataset = DQAImpressionDataset(parquet_path, emb_map, indices=val_idx)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
            # The PAL embedding is capped at MAX_POSITION; LR still uses raw positions below.
            pal_pos = pos.clamp(max=model_pal.position_embedding.num_embeddings - 1)
            col = batch["is_left_col"].to(device)
            labels = batch["label"].to(device)
            
            inputs = {
                "query_emb": q_emb, "item_emb": i_emb, 
                "position": pal_pos, "is_left_col": col
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
            X_lr = np.nan_to_num(
                X_lr.astype(np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
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

def build_learned_bias_table(positions, p_e_std, p_e_dqa):
    return [
        {
            "Position": int(position),
            "Standard P(E)": float(std_value),
            "DQA P(E)": float(dqa_value),
        }
        for position, std_value, dqa_value in zip(positions, p_e_std, p_e_dqa)
    ]


def save_learned_bias_table(rows, csv_path=LEARNED_BIAS_CSV, tex_path=LEARNED_BIAS_TEX):
    headers = ["Position", "Standard P(E)", "DQA P(E)"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["Position"],
                f"{row['Standard P(E)']:.6f}",
                f"{row['DQA P(E)']:.6f}",
            ])

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Learned examination bias values extracted from Figure 4.}",
        "\\label{tab:learned-bias-values}",
        "\\begin{tabular}{rcc}",
        "\\hline",
        "Position & Standard $P(E)$ & DQA $P(E)$ \\\\",
        "\\hline",
    ]
    for row in rows:
        latex_lines.append(
            f"{row['Position']} & {row['Standard P(E)']:.4f} & {row['DQA P(E)']:.4f} \\\\"
        )
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print("\nLearned bias values table")
    print("-" * 52)
    print(f"{headers[0]:<10} | {headers[1]:<14} | {headers[2]:<10}")
    print("-" * 52)
    for row in rows:
        print(
            f"{row['Position']:<10} | "
            f"{row['Standard P(E)']:<14.6f} | "
            f"{row['DQA P(E)']:<10.6f}"
        )
    print("-" * 52)
    print(f"Saved learned bias values to {csv_path} and {tex_path}")


def extract_and_plot_curves(pal_std, pal_dqa, emb_map, device):
    print("Extracting internal P(E) representations for Figure 4...")
    pal_std.eval()
    pal_dqa.eval()

    with torch.no_grad():
        # 1. Dummy Layout Setup: positions 1 through 10 in the two-column layout.
        # Odd ranks are left-column results; even ranks are right-column results.
        pos_dummy = torch.arange(1, 11, dtype=torch.long).unsqueeze(1).to(device) # Shape (10, 1)
        col_dummy = (pos_dummy % 2 != 0).float().to(device)             # Shape (10, 1)

        # 2. Extract Standard P(E)
        pos_emb_std = pal_std.position_embedding(pos_dummy.squeeze(-1))
        exam_in_std = torch.cat([pos_emb_std, col_dummy], dim=-1)
        p_e_std = pal_std.examination_tower(exam_in_std).cpu().numpy().flatten()

        # 3. Extract DQA P(E)
        # Get the average semantic space of a DQA response
        all_dqa_vecs = list(emb_map['dqa'].values())
        avg_dqa_vec = np.mean(all_dqa_vecs, axis=0)
        dqa_tensor = torch.tensor(avg_dqa_vec, dtype=torch.float32).unsqueeze(0).repeat(10, 1).to(device)

        pos_emb_dqa = pal_dqa.position_embedding(pos_dummy.squeeze(-1))
        exam_in_dqa = torch.cat([pos_emb_dqa, col_dummy, dqa_tensor], dim=-1)
        p_e_dqa = pal_dqa.examination_tower(exam_in_dqa).cpu().numpy().flatten()

    positions = pos_dummy.cpu().numpy().flatten()
    learned_bias_table = build_learned_bias_table(positions, p_e_std, p_e_dqa)

    # 4. Generate the Thesis Graph
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(1, 11), p_e_std, marker='o', markersize=8, 
             linestyle='-', linewidth=2, color='#008080', label='Learned: Standard Layout')
    plt.plot(range(1, 11), p_e_dqa, marker='s', markersize=8, 
             linestyle='--', linewidth=2, color='#D11141', label='Learned: DQA-Present')

    plt.title("Learned Examination Bias Under the Two-Column Layout", fontsize=16, pad=20)
    plt.xlabel("Organic Rank Position", fontsize=12)
    plt.ylabel("Learned Examination Probability $P(E)$", fontsize=12)
    plt.xticks(range(1, 11))
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save exactly as required for the thesis LaTeX
    file_name = 'Fig_4_learned_bias_true_layout.pdf'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f"Saved learned bias curve to {file_name}")
    save_learned_bias_table(learned_bias_table)
    return learned_bias_table

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluation running on: {device}")

    # 1. Load Embeddings and Datasets
    emb_map = load_embedding_map("data/embeddings_map.pkl")
    std_test_path = "data/test_merged_text.parquet"
    dqa_path = "data/dqa_merged_text.parquet"

    loader_std = get_test_loader_standard(std_test_path, emb_map)
    loader_dqa = get_val_loader_dqa(dqa_path, emb_map)

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
