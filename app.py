"""
Personality Predictor API
FastAPI + Pydantic — all preprocessing inside sklearn Pipeline (no training-serving skew)
"""
import logging
import time
from typing import Literal
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Personality Predictor API",
    description="Predicts Introvert / Extrovert from behavioural features.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Load model once at startup ────────────────────────────────────────────────
model    = joblib.load("personality_model.pkl")
FEATURES = joblib.load("feature_names.pkl")
logger.info(f"Model loaded. Features expected: {FEATURES}")

# ── Request / Response schemas ────────────────────────────────────────────────
class PersonalityInput(BaseModel):
    Time_spent_Alone:          float = Field(..., ge=0, le=11,  description="Hours alone per day (0–11)")
    Stage_fear:                Literal["Yes", "No"]
    Social_event_attendance:   float = Field(..., ge=0, le=10,  description="Social event frequency (0–10)")
    Going_outside:             float = Field(..., ge=0, le=7,   description="Times outside per week (0–7)")
    Drained_after_socializing: Literal["Yes", "No"]
    Friends_circle_size:       float = Field(..., ge=0, le=15,  description="Close friends count (0–15)")
    Post_frequency:            float = Field(..., ge=0, le=10,  description="Social media posts/week (0–10)")

    class Config:
        json_schema_extra = {"example": {
            "Time_spent_Alone": 7.0, "Stage_fear": "Yes",
            "Social_event_attendance": 2.0, "Going_outside": 2.0,
            "Drained_after_socializing": "Yes",
            "Friends_circle_size": 3.0, "Post_frequency": 2.0
        }}

class PredictionResponse(BaseModel):
    prediction:    str
    confidence:    float
    probabilities: dict
    model_version: str = "1.0.0"

# ── Middleware: log every request + latency ────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.1f}ms)")
    return response

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Personality Predictor API", "status": "online",
            "version": "1.0.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PersonalityInput):
    try:
        # Build dataframe in the exact feature order the pipeline expects
        row = pd.DataFrame([data.model_dump()])[FEATURES]
        logger.info(f"Inference request: {row.to_dict(orient='records')[0]}")

        pred  = model.predict(row)[0]
        proba = model.predict_proba(row)[0]
        label = "Extrovert" if pred == 1 else "Introvert"

        result = {
            "prediction":    label,
            "confidence":    round(float(max(proba)), 4),
            "probabilities": {
                "Introvert": round(float(proba[0]), 4),
                "Extrovert": round(float(proba[1]), 4)
            },
            "model_version": "1.0.0"
        }
        logger.info(f"Prediction: {label}  confidence={result['confidence']}")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
