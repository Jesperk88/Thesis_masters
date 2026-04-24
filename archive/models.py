import torch
import torch.nn as nn

class PALModel(nn.Module):
    """
    Position-Aware Learning (PAL) Model.
    Decouples examination probability from item relevance.
    """
    def __init__(self, max_position=50, use_dqa_feature=False):
        super(PALModel, self).__init__()
        self.use_dqa_feature = use_dqa_feature
        
        # --- Examination Pathway ---
        # 1. Embed the integer position into a dense vector
        self.pos_embedding = nn.Embedding(num_embeddings=max_position + 1, embedding_dim=4)
        
        # 2. Determine input size: 4 (pos) + 1 (left_col) + [1 if DQA else 0]
        exam_input_dim = 4 + 1 + (1 if use_dqa_feature else 0)
        
        # 3. Shallow MLP to learn the examination bias
        self.exam_mlp = nn.Sequential(
            nn.Linear(exam_input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() # Outputs P(E) between 0 and 1
        )
        
        # --- Relevance Pathway ---
        # Because we are not using content features (per the methodology),
        # relevance is modeled as a single global learnable parameter.
        self.global_relevance = nn.Parameter(torch.tensor([0.0]))

    def forward(self, position, is_left, dqa_present=None):
        # 1. Calculate Examination Probability P(E)
        pos_emb = self.pos_embedding(position)
        
        features = [pos_emb, is_left.unsqueeze(1).float()]
        if self.use_dqa_feature and dqa_present is not None:
            features.append(dqa_present.unsqueeze(1).float())
            
        exam_input = torch.cat(features, dim=1)
        prob_exam = self.exam_mlp(exam_input)
        
        # 2. Calculate Relevance Probability P(R)
        prob_rel = torch.sigmoid(self.global_relevance)
        
        # 3. Final Prediction: P(Click) = P(E) * P(R)
        prob_click = prob_exam * prob_rel
        
        return prob_click.squeeze()