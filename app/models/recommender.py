"""Two-tower recommendation model for same-side matching.

Both towers share the same embedding table since all users are in one pool.
The model predicts P(match | viewer, candidate) using learned user embeddings
combined with handcrafted features.
"""

import torch
import torch.nn as nn


class TrainingModel(nn.Module):
    """Full training model with dropout, regularization, and loss computation."""

    def __init__(self, num_users: int, embedding_dim: int = 32, num_features: int = 8, dropout: float = 0.2):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.feature_encoder = nn.Linear(num_features, embedding_dim)

        self.interaction = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),  # viewer_emb + viewer_feat + cand_emb + cand_feat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        for module in self.interaction:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, viewer_id, candidate_id, viewer_features, candidate_features):
        v_emb = self.user_embedding(viewer_id)
        c_emb = self.user_embedding(candidate_id)
        v_feat = self.feature_encoder(viewer_features)
        c_feat = self.feature_encoder(candidate_features)

        combined = torch.cat([v_emb, v_feat, c_emb, c_feat], dim=1)
        logits = self.interaction(combined).squeeze(-1)
        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)


class InferenceModel(nn.Module):
    """Stripped-down inference model. No dropout, no loss, just scores."""

    def __init__(self, num_users: int, embedding_dim: int = 32, num_features: int = 8):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.feature_encoder = nn.Linear(num_features, embedding_dim)

        self.interaction = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, viewer_id, candidate_id, viewer_features, candidate_features):
        v_emb = self.user_embedding(viewer_id)
        c_emb = self.user_embedding(candidate_id)
        v_feat = self.feature_encoder(viewer_features)
        c_feat = self.feature_encoder(candidate_features)

        combined = torch.cat([v_emb, v_feat, c_emb, c_feat], dim=1)
        logits = self.interaction(combined).squeeze(-1)
        return torch.sigmoid(logits)


# Weight mapping: training param name → inference param name
WEIGHT_MAPPING = {
    "user_embedding.weight": "user_embedding.weight",
    "feature_encoder.weight": "feature_encoder.weight",
    "feature_encoder.bias": "feature_encoder.bias",
    # interaction layers: skip dropout layers (indices shift)
    # training: Linear(0), ReLU(1), Dropout(2), Linear(3), ReLU(4), Dropout(5), Linear(6)
    # inference: Linear(0), ReLU(1), Linear(2), ReLU(3), Linear(4)
    "interaction.0.weight": "interaction.0.weight",
    "interaction.0.bias": "interaction.0.bias",
    "interaction.3.weight": "interaction.2.weight",
    "interaction.3.bias": "interaction.2.bias",
    "interaction.6.weight": "interaction.4.weight",
    "interaction.6.bias": "interaction.4.bias",
}


def load_training_weights_into_inference(
    inference_model: InferenceModel,
    training_checkpoint: dict,
) -> InferenceModel:
    """Load trained weights into the inference model using the weight mapping."""
    inference_state = inference_model.state_dict()

    for train_name, inf_name in WEIGHT_MAPPING.items():
        if train_name in training_checkpoint:
            inference_state[inf_name] = training_checkpoint[train_name]

    inference_model.load_state_dict(inference_state)
    return inference_model
