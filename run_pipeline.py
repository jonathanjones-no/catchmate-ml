"""Standalone pipeline runner for Railway cron.

Usage: python run_pipeline.py
"""

import sys
from app.train import train_model
from app.predict import predict_all_pairs

print("Starting ML pipeline...")

train_result = train_model()
print(f"Train: {train_result}")

if train_result["status"] == "success":
    predict_result = predict_all_pairs()
    print(f"Predict: {predict_result}")
else:
    print(f"Training skipped: {train_result.get('reason', 'unknown')}")

# Exit with non-zero if training failed (not skipped)
if train_result["status"] not in ("success", "skipped"):
    sys.exit(1)
