"""Basic tests for the recommendation model."""

import torch
import pytest
from app.models.recommender import (
    TrainingModel,
    InferenceModel,
    load_training_weights_into_inference,
)


@pytest.fixture
def training_model():
    return TrainingModel(num_users=100, embedding_dim=16, num_features=8)


@pytest.fixture
def inference_model():
    return InferenceModel(num_users=100, embedding_dim=16, num_features=8)


def test_training_model_forward(training_model):
    viewers = torch.tensor([1, 2, 3])
    candidates = torch.tensor([4, 5, 6])
    v_feats = torch.randn(3, 8)
    c_feats = torch.randn(3, 8)

    logits = training_model(viewers, candidates, v_feats, c_feats)
    assert logits.shape == (3,)


def test_inference_model_forward(inference_model):
    viewers = torch.tensor([1, 2, 3])
    candidates = torch.tensor([4, 5, 6])
    v_feats = torch.randn(3, 8)
    c_feats = torch.randn(3, 8)

    scores = inference_model(viewers, candidates, v_feats, c_feats)
    assert scores.shape == (3,)
    assert (scores >= 0).all() and (scores <= 1).all()  # sigmoid output


def test_inference_outputs_probabilities(inference_model):
    viewers = torch.tensor([1])
    candidates = torch.tensor([2])
    v_feats = torch.randn(1, 8)
    c_feats = torch.randn(1, 8)

    score = inference_model(viewers, candidates, v_feats, c_feats)
    assert 0.0 <= score.item() <= 1.0


def test_weight_transfer(training_model, inference_model):
    # Train for a few steps so weights diverge from init
    optimizer = torch.optim.Adam(training_model.parameters(), lr=0.01)
    for _ in range(5):
        viewers = torch.randint(1, 100, (10,))
        candidates = torch.randint(1, 100, (10,))
        v_feats = torch.randn(10, 8)
        c_feats = torch.randn(10, 8)
        labels = torch.rand(10)

        logits = training_model(viewers, candidates, v_feats, c_feats)
        loss = training_model.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Transfer weights
    load_training_weights_into_inference(inference_model, training_model.state_dict())

    # Both models should produce similar outputs (not identical due to dropout in training)
    training_model.eval()
    inference_model.eval()

    test_v = torch.tensor([1, 2])
    test_c = torch.tensor([3, 4])
    test_vf = torch.randn(2, 8)
    test_cf = torch.randn(2, 8)

    with torch.no_grad():
        train_out = torch.sigmoid(training_model(test_v, test_c, test_vf, test_cf))
        infer_out = inference_model(test_v, test_c, test_vf, test_cf)

    assert torch.allclose(train_out, infer_out, atol=1e-6)


def test_training_loss_decreases(training_model):
    optimizer = torch.optim.Adam(training_model.parameters(), lr=0.01)
    losses = []

    for _ in range(20):
        # Fixed data so loss should decrease
        viewers = torch.tensor([1, 2, 3, 4, 5])
        candidates = torch.tensor([6, 7, 8, 9, 10])
        v_feats = torch.randn(5, 8)
        c_feats = torch.randn(5, 8)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

        logits = training_model(viewers, candidates, v_feats, c_feats)
        loss = training_model.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # Loss should generally decrease
    assert losses[-1] < losses[0]


def test_same_side_symmetry(inference_model):
    """In same-side matching, the model should handle any user as viewer or candidate."""
    user_a = torch.tensor([5])
    user_b = torch.tensor([10])
    feats_a = torch.randn(1, 8)
    feats_b = torch.randn(1, 8)

    with torch.no_grad():
        score_ab = inference_model(user_a, user_b, feats_a, feats_b)
        score_ba = inference_model(user_b, user_a, feats_b, feats_a)

    # Scores don't need to be equal (asymmetric matching is valid)
    # but both should be valid probabilities
    assert 0.0 <= score_ab.item() <= 1.0
    assert 0.0 <= score_ba.item() <= 1.0
