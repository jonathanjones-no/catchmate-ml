"""FastAPI service for CatchMate ML recommendation system."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .config import settings
from .train import train_model
from .predict import predict_all_pairs, get_latest_model_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate required config on startup."""
    missing = []
    if not settings.database_url.get_secret_value():
        missing.append("DATABASE_URL")
    if not settings.supabase_url:
        missing.append("SUPABASE_URL")
    if not settings.supabase_service_key.get_secret_value():
        missing.append("SUPABASE_SERVICE_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    yield


app = FastAPI(title="CatchMate ML", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    model_path = get_latest_model_path()
    return {
        "status": "healthy",
        "model_loaded": model_path is not None,
        "model_path": model_path,
    }


@app.post("/train")
async def train():
    try:
        result = train_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict():
    try:
        result = predict_all_pairs()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline")
async def pipeline():
    """Run full pipeline: train then predict."""
    try:
        train_result = train_model()
        if train_result["status"] != "success":
            return {"train": train_result, "predict": {"status": "skipped"}}

        predict_result = predict_all_pairs()
        return {"train": train_result, "predict": predict_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
