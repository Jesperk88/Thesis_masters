"""
models_final.py
---------------
Dual-Tower Position-Aware Learning (PAL) architecture.

Tower 1 — Relevance Pathway:
    Ingests concatenated query + item embeddings (1536-dim).
    Learns semantic relevance P(R) independently of position.

Tower 2 — Examination Pathway:
    Ingests position embedding + column indicator.
    When use_dqa=True, also ingests the DQA output embedding
    to model how the answer's semantic quality suppresses
    user scrolling attention (Good Abandonment).

Final CTR prediction: P(C) = P(E) * P(R)
"""

import torch
import torch.nn as nn

EMBEDDING_DIM    = 768
POSITION_EMB_DIM = 4        # Same as original PAL methodology
MAX_POSITION     = 60       # Global maximum observed position


class DualTowerPAL(nn.Module):
    """
    Dual-Tower PAL model.

    Args:
        use_dqa (bool): If True, the examination tower also ingests
                        the DQA output embedding. Set False for the
                        standard baseline, True for the DQA-aware model.
        max_position (int): Maximum position index for the embedding layer.
    """

    def __init__(self, use_dqa: bool = False, max_position: int = MAX_POSITION):
        super().__init__()
        self.use_dqa = use_dqa

        # ------------------------------------------------------------------
        # Tower 1: Relevance Pathway
        # Input: [query_emb (768) || item_emb (768)] = 1536-dim
        # Output: scalar P(R) in (0, 1)
        # ------------------------------------------------------------------
        self.relevance_tower = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Dropout prevents overfitting to static vectors
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Tower 2: Examination Pathway
        # Position embedding (learned, 4-dim) + column indicator (1-dim)
        # If use_dqa: also concatenate DQA embedding (768-dim)
        # Output: scalar P(E) in (0, 1)
        # ------------------------------------------------------------------
        self.position_embedding = nn.Embedding(
            num_embeddings=max_position + 1,
            embedding_dim=POSITION_EMB_DIM,
        )

        exam_input_dim = POSITION_EMB_DIM + 1  # position emb + column indicator
        if use_dqa:
            exam_input_dim += EMBEDDING_DIM     # + DQA embedding

        self.examination_tower = nn.Sequential(
            nn.Linear(exam_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dictionary from the Dataset __getitem__, containing
                   query_emb, item_emb, position, is_left_col,
                   and optionally dqa_emb.
        Returns:
            predicted CTR as a (batch_size, 1) tensor.
        """
        # --- Relevance Tower ---
        relevance_input = torch.cat(
            [batch["query_emb"], batch["item_emb"]], dim=-1
        )
        p_relevance = self.relevance_tower(relevance_input)  # (B, 1)

        # --- Examination Tower ---
        pos_emb = self.position_embedding(
            batch["position"].squeeze(-1)
        )                                                    # (B, 4)

        col = batch["is_left_col"]                           # (B, 1)

        if self.use_dqa:
            exam_input = torch.cat(
                [pos_emb, col, batch["dqa_emb"]], dim=-1
            )
        else:
            exam_input = torch.cat([pos_emb, col], dim=-1)

        p_examination = self.examination_tower(exam_input)   # (B, 1)

        # --- PAL Merge: P(C) = P(E) * P(R) ---
        return p_examination * p_relevance                   # (B, 1)