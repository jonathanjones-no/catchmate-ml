"""Training pipeline: fetch data, train model, save checkpoint."""

import os
import time
import torch
import numpy as np
from .config import settings
from .db import fetch_training_data
from .models.recommender import TrainingModel, InferenceModel, load_training_weights_into_inference


def build_user_index(users: list[dict]) -> dict:
    """Map user UUIDs to integer indices for embedding lookup."""
    return {u["id"]: i + 1 for i, u in enumerate(users)}  # 0 reserved for padding


NUM_FEATURES = 12


def build_feature_vector(user: dict) -> list[float]:
    """Convert user dict to a fixed-size feature vector."""
    return [
        1.0 if user.get("is_verified") else 0.0,
        1.0 if user.get("has_photo") else 0.0,
        1.0 if user.get("has_bio") else 0.0,
        min(len(user.get("sport_types") or []) / 5.0, 1.0),  # sport count normalized
        min((user.get("age") or 25) / 40.0, 1.0),  # age normalized
        (user.get("lat") or 39.7) / 90.0,  # lat normalized (from PostGIS)
        ((user.get("lng") or -105.0) + 180.0) / 360.0,  # lng normalized
        max(0, 1.0 - (user.get("days_old") or 0) / 30.0),  # account recency
        max(0, 1.0 - (user.get("days_inactive") or 0) / 30.0),  # activity recency
        min((user.get("intrinsic_base") or 0) / 100.0, 1.0),  # pre-computed intrinsic
        min((user.get("initiative") or 0), 1.0),  # pre-computed initiative
        min((user.get("total_interactions") or 0) / 50.0, 1.0),  # interaction count normalized
    ]


def prepare_training_pairs(data: dict, user_index: dict, user_features: dict):
    """Build training pairs from swipe data.

    Positive: mutual right swipes (matches)
    Negative: left swipes + unreciprocated right swipes
    """
    viewers, candidates, v_feats, c_feats, labels = [], [], [], [], []

    # Positive pairs from matches
    for match in data["matches"]:
        u1 = match["user1_id"]
        u2 = match["user2_id"]
        if u1 in user_index and u2 in user_index:
            # Both directions are positive
            viewers.append(user_index[u1])
            candidates.append(user_index[u2])
            v_feats.append(user_features.get(u1, [0.0] * NUM_FEATURES))
            c_feats.append(user_features.get(u2, [0.0] * NUM_FEATURES))
            labels.append(1.0)

            viewers.append(user_index[u2])
            candidates.append(user_index[u1])
            v_feats.append(user_features.get(u2, [0.0] * NUM_FEATURES))
            c_feats.append(user_features.get(u1, [0.0] * NUM_FEATURES))
            labels.append(1.0)

    # Negative pairs from left swipes
    for swipe in data["swipes"]:
        if swipe["direction"] == "left":
            s = swipe["swiper_id"]
            d = swipe["swiped_id"]
            if s in user_index and d in user_index:
                viewers.append(user_index[s])
                candidates.append(user_index[d])
                v_feats.append(user_features.get(s, [0.0] * NUM_FEATURES))
                c_feats.append(user_features.get(d, [0.0] * NUM_FEATURES))
                labels.append(0.0)

    return {
        "viewers": torch.tensor(viewers, dtype=torch.long),
        "candidates": torch.tensor(candidates, dtype=torch.long),
        "viewer_features": torch.tensor(v_feats, dtype=torch.float32),
        "candidate_features": torch.tensor(c_feats, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


def train_model(epochs: int = 50) -> dict:
    """Full training pipeline. Returns metadata about the training run."""
    start_time = time.time()

    # Fetch data
    data = fetch_training_data()
    users = data["users"]

    if len(users) < 10 or len(data["swipes"]) < 20:
        return {
            "status": "skipped",
            "reason": "insufficient data",
            "users": len(users),
            "swipes": len(data["swipes"]),
        }

    # Build indices and features
    user_index = build_user_index(users)
    user_features = {u["id"]: build_feature_vector(u) for u in users}

    # Prepare training data
    pairs = prepare_training_pairs(data, user_index, user_features)

    if len(pairs["labels"]) < 10:
        return {"status": "skipped", "reason": "insufficient training pairs"}

    # Initialize model
    num_users = len(user_index) + 1  # +1 for padding index
    model = TrainingModel(
        num_users=num_users,
        embedding_dim=settings.embedding_dim,
        num_features=NUM_FEATURES,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    # Training loop
    model.train()
    losses = []
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(len(pairs["labels"]))
        for key in pairs:
            pairs[key] = pairs[key][perm]

        # Mini-batch training
        for i in range(0, len(pairs["labels"]), settings.batch_size):
            end = min(i + settings.batch_size, len(pairs["labels"]))
            batch_viewers = pairs["viewers"][i:end]
            batch_candidates = pairs["candidates"][i:end]
            batch_v_feats = pairs["viewer_features"][i:end]
            batch_c_feats = pairs["candidate_features"][i:end]
            batch_labels = pairs["labels"][i:end]

            optimizer.zero_grad()
            logits = model(batch_viewers, batch_candidates, batch_v_feats, batch_c_feats)
            loss = model.compute_loss(logits, batch_labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    # Save training checkpoint
    os.makedirs(settings.model_dir, exist_ok=True)
    model_version = f"v_{int(time.time())}"
    checkpoint_path = os.path.join(settings.model_dir, f"{model_version}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "user_index": user_index,
        "user_features": user_features,
        "model_version": model_version,
        "num_users": num_users,
        "embedding_dim": settings.embedding_dim,
    }, checkpoint_path)

    # Build inference model
    inference_model = InferenceModel(
        num_users=num_users,
        embedding_dim=settings.embedding_dim,
        num_features=NUM_FEATURES,
    )
    load_training_weights_into_inference(inference_model, model.state_dict())

    inference_path = os.path.join(settings.model_dir, f"{model_version}_inference.pt")
    torch.save({
        "model_state_dict": inference_model.state_dict(),
        "user_index": user_index,
        "user_features": user_features,
        "model_version": model_version,
        "num_users": num_users,
        "embedding_dim": settings.embedding_dim,
    }, inference_path)

    elapsed = time.time() - start_time

    return {
        "status": "success",
        "model_version": model_version,
        "checkpoint_path": checkpoint_path,
        "inference_path": inference_path,
        "training_pairs": len(pairs["labels"]),
        "users": len(users),
        "epochs": epochs,
        "final_loss": float(np.mean(losses[-10:])) if losses else None,
        "elapsed_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    result = train_model()
    print(result)
