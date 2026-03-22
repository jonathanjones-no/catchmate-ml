# catchmate-ml

Recommendation and matching ML service for [CatchMate](https://github.com/jonathanjones-no/catchmate). Scores user pairs to predict match compatibility, powering the Discover feed ranking algorithm.

## Architecture

```
┌──────────────────────────┐     ┌─────────────────────┐
│  CatchMate App           │     │  Supabase Postgres  │
│  (React Native)          │     │                     │
│          │               │     │  users, swipes,     │
│          ▼               │     │  matches, sessions, │
│  get_discover_profiles() │────▶│  analytics_events   │
│  (Supabase RPC)          │     │         │           │
│          │               │     │         ▼           │
│          ▼               │     │  user_pair_scores   │
│  Ranked deck             │     │  (ML predictions)   │
└──────────────────────────┘     └─────────┬───────────┘
                                           ▲
                                  ┌────────┘
                      ┌───────────┴──────────────┐
                      │  catchmate-ml            │
                      │  (this service)          │
                      │                          │
                      │  Nightly batch:          │
                      │  1. Pull behavioral data │
                      │  2. Train model          │
                      │  3. Score all pairs      │
                      │  4. Write predictions    │
                      └──────────────────────────┘
```

## Model

Two-tower embedding architecture (same-side matching — both towers share the embedding table):

- **Input:** User features (sports, age, location, verification, behavioral signals) + swipe/match/session history
- **Embedding:** Learned user representations (32-64 dim)
- **Interaction:** MLP on concatenated viewer + candidate embeddings
- **Output:** Predicted match probability (0-1)

Separate training and inference models:
- **Training model:** Includes dropout, regularization, loss computation
- **Inference model:** Stripped down, outputs scores only

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/train` | POST | Trigger model training on latest data |
| `/predict` | POST | Batch score all eligible user pairs |
| `/metrics` | GET | Model performance metrics |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Configure Supabase connection
```

## Development

```bash
# Run locally
uvicorn app.main:app --reload --port 8000

# Run tests
pytest

# Train model locally
python -m app.train
```

## Deployment

Deployed on Railway/Fly.io. Nightly cron triggers `/train` → `/predict` pipeline.

## Environment Variables

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Production Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Service role key (reads all tables) |
| `DATABASE_URL` | Direct Postgres connection string |
| `MODEL_DIR` | Directory for model checkpoints (default: `./checkpoints`) |
| `EMBEDDING_DIM` | User embedding dimension (default: 32) |
| `BATCH_SIZE` | Training batch size (default: 256) |
| `LEARNING_RATE` | Training learning rate (default: 0.001) |
