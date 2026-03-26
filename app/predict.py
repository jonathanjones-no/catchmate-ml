"""Batch prediction pipeline: load model, score all pairs, write to DB."""

import os
import glob
import torch
from .config import settings
from .db import fetch_eligible_pairs, write_pair_scores, get_connection, fetch_behavioral_signals
from .models.recommender import InferenceModel
from .train import build_feature_vector, NUM_FEATURES


def get_latest_model_path() -> str | None:
    """Find the most recent inference model checkpoint."""
    pattern = os.path.join(settings.model_dir, "*_inference.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def predict_all_pairs() -> dict:
    """Score all eligible user pairs and write to user_pair_scores."""
    model_path = get_latest_model_path()
    if not model_path:
        return {"status": "skipped", "reason": "no trained model found"}

    # Load model
    checkpoint = torch.load(model_path, weights_only=False)
    user_index = checkpoint["user_index"]
    user_features_map = checkpoint["user_features"]
    model_version = checkpoint["model_version"]

    model = InferenceModel(
        num_users=checkpoint["num_users"],
        embedding_dim=checkpoint["embedding_dim"],
        num_features=NUM_FEATURES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Fetch eligible pairs
    pairs = fetch_eligible_pairs()
    if not pairs:
        return {"status": "skipped", "reason": "no eligible pairs"}

    # Refresh user features from DB + BigQuery behavioral signals
    behavioral_signals = fetch_behavioral_signals()
    conn = get_connection()
    try:
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT u.id, u.sport_types, u.age, u.is_verified,
                       u.photo_url IS NOT NULL as has_photo,
                       u.bio IS NOT NULL AND u.bio != '' as has_bio,
                       ST_Y(u.location::geometry) as lat,
                       ST_X(u.location::geometry) as lng,
                       EXTRACT(EPOCH FROM (now() - u.created_at)) / 86400 as days_old,
                       EXTRACT(EPOCH FROM (now() - u.last_active_at)) / 86400 as days_inactive,
                       COALESCE(uis.intrinsic_base, 0) as intrinsic_base,
                       COALESCE(uis.initiative, 0) as initiative,
                       COALESCE(uis.total_interactions, 0) as total_interactions
                FROM public.users u
                LEFT JOIN public.user_intrinsic_scores uis ON uis.user_id = u.id
                WHERE u.first_name IS NOT NULL AND u.first_name != ''
                  AND u.is_test_account = false
            """)
            for row in cur.fetchall():
                user_features_map[row["id"]] = build_feature_vector(
                    row, behavioral_signals.get(row["id"])
                )
    finally:
        conn.close()

    # Batch predict
    scores = []
    default_features = [0.0] * NUM_FEATURES
    batch_size = 1024

    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]

            viewer_ids = []
            candidate_ids = []
            v_feats = []
            c_feats = []

            for pair in batch:
                vid = pair["viewer_id"]
                cid = pair["candidate_id"]

                # Skip pairs where either user isn't in the model's index
                # (new users since last training)
                v_idx = user_index.get(vid)
                c_idx = user_index.get(cid)

                if v_idx is None or c_idx is None:
                    # Default score for unknown users
                    scores.append({
                        "viewer_id": vid,
                        "candidate_id": cid,
                        "score": 0.5,
                        "model_version": model_version,
                    })
                    continue

                viewer_ids.append(v_idx)
                candidate_ids.append(c_idx)
                v_feats.append(user_features_map.get(vid, default_features))
                c_feats.append(user_features_map.get(cid, default_features))

            if viewer_ids:
                preds = model(
                    torch.tensor(viewer_ids, dtype=torch.long),
                    torch.tensor(candidate_ids, dtype=torch.long),
                    torch.tensor(v_feats, dtype=torch.float32),
                    torch.tensor(c_feats, dtype=torch.float32),
                )

                for j, pair in enumerate([b for b in batch if user_index.get(b["viewer_id"]) is not None and user_index.get(b["candidate_id"]) is not None]):
                    scores.append({
                        "viewer_id": pair["viewer_id"],
                        "candidate_id": pair["candidate_id"],
                        "score": float(preds[j]),
                        "model_version": model_version,
                    })

    # Write to DB
    write_pair_scores(scores)

    return {
        "status": "success",
        "model_version": model_version,
        "pairs_scored": len(scores),
        "pairs_default": sum(1 for s in scores if s["score"] == 0.5),
    }


if __name__ == "__main__":
    result = predict_all_pairs()
    print(result)
