"""
Credit Card Fraud Detection — REST API
========================================
FastAPI application serving the optimized Random Forest model.
Model is loaded from MLflow registry (with joblib fallback).
Includes input validation, feature engineering, rate limiting, and Swagger docs.

Run:
    uvicorn api.main:app --reload
"""

import os
import json
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler

from api.schemas import PredictRequest, PredictResponse, HealthResponse

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

BEST_THRESHOLD = 0.5  # Default; loaded from model_config.json at startup
MODEL_VERSION = "unknown"  # Resolved dynamically from MLflow registry at startup
REGISTRY_NAME = "CreditCardFraudDetector"

COLS_TO_SCALE = [
    "Amount", "Time", "Time_in_day", "Amount_log",
    "Time_Amount", "Time_Amount_sq", "Amount_per_Time",
]

# Feature order must match training data exactly (from parquet column order)
FEATURE_ORDER = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "Time_in_day", "Amount_log", "Time_Amount", "Time_Amount_sq", "Amount_per_Time",
]

# Rate limiting: max requests per IP per window
RATE_LIMIT_MAX = 100
RATE_LIMIT_WINDOW = 60  # seconds

logger = logging.getLogger("fraud_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ──────────────────────────────────────────────────────────────────────
# Global state (populated at startup)
# ──────────────────────────────────────────────────────────────────────

model = None
scaler: StandardScaler | None = None


# ──────────────────────────────────────────────────────────────────────
# Model & scaler loading
# ──────────────────────────────────────────────────────────────────────

def load_model():
    """Load model from MLflow registry, falling back to joblib."""
    try:
        import mlflow  # type: ignore

        tracking_uri = f"file:///{MLRUNS_DIR.replace(os.sep, '/')}"
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{REGISTRY_NAME}@production"
        loaded = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded from MLflow registry: %s@production", REGISTRY_NAME)
        return loaded
    except Exception as exc:
        logger.warning("MLflow load failed (%s), falling back to joblib.", exc)
        path = os.path.join(MODEL_DIR, "optimized_model.joblib")
        loaded = joblib.load(path)
        logger.info("Model loaded from joblib: %s", path)
        return loaded


def fit_scaler() -> StandardScaler:
    """Fit StandardScaler on the original (pre-SMOTE) training data."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

    # Try loading a pre-saved scaler first
    if os.path.exists(scaler_path):
        s = joblib.load(scaler_path)
        logger.info("Scaler loaded from %s", scaler_path)
        return s

    # Fit from training data
    train = pd.read_parquet(os.path.join(DATA_DIR, "train_original.parquet"))
    s = StandardScaler()
    s.fit(train[COLS_TO_SCALE])
    # Persist for future loads
    joblib.dump(s, scaler_path)
    logger.info("Scaler fitted and saved to %s", scaler_path)
    return s


def load_model_config() -> dict:
    """Load model config (threshold, best model info, registry version)."""
    config_path = os.path.join(MODEL_DIR, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        logger.info("Model config loaded from %s", config_path)
        return config
    logger.warning("model_config.json not found, using defaults")
    return {}


def load_registry_version() -> str:
    """Load the current production model version from MLflow registry."""
    try:
        import mlflow  # type: ignore
        tracking_uri = f"file:///{MLRUNS_DIR.replace(os.sep, '/')}"
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(REGISTRY_NAME, "production")
        version = str(mv.version)
        logger.info("MLflow registry production version: %s", version)
        return version
    except Exception as exc:
        logger.warning("Could not read MLflow registry version: %s", exc)
        return MODEL_VERSION


# ──────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors preprocessing.ipynb)
# ──────────────────────────────────────────────────────────────────────

def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Accept raw V1-V28 + Time + Amount, compute 5 engineered features,
    scale the appropriate columns, and return a DataFrame in FEATURE_ORDER.
    """
    row = dict(raw)

    # Engineered features
    row["Time_in_day"] = row["Time"] % 86400
    row["Amount_log"] = np.log1p(row["Amount"])
    row["Time_Amount"] = row["Time"] * row["Amount"]
    row["Time_Amount_sq"] = (row["Time"] * row["Amount"]) ** 2
    amount_per_time = row["Amount"] / row["Time"] if row["Time"] != 0 else 0.0
    row["Amount_per_Time"] = amount_per_time

    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    # Scale
    df[COLS_TO_SCALE] = scaler.transform(df[COLS_TO_SCALE])

    return df.values


# ──────────────────────────────────────────────────────────────────────
# Rate limiter (in-memory sliding window)
# ──────────────────────────────────────────────────────────────────────

_request_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate limit exceeded."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Prune old entries
    timestamps = _request_log[client_ip]
    _request_log[client_ip] = [t for t in timestamps if t > window_start]

    if len(_request_log[client_ip]) >= RATE_LIMIT_MAX:
        return False

    _request_log[client_ip].append(now)
    return True


# ──────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, scaler, threshold, and version at startup."""
    global model, scaler, BEST_THRESHOLD, MODEL_VERSION
    model = load_model()
    scaler = fit_scaler()

    config = load_model_config()
    BEST_THRESHOLD = float(config.get("best_threshold", 0.5))
    MODEL_VERSION = load_registry_version()

    logger.info(
        "API ready — model v%s, threshold %.4f loaded.",
        MODEL_VERSION, BEST_THRESHOLD,
    )
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "REST API serving an optimized Random Forest model for credit card "
        "fraud detection. Includes feature engineering, input validation, "
        "and rate limiting."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────
# Rate-limit middleware
# ──────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )
    return await call_next(request)


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", model_version=MODEL_VERSION)


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(body: PredictRequest):
    """
    Predict whether a credit card transaction is fraudulent.

    Accepts raw transaction features (V1-V28, Amount, Time).
    Returns fraud probability, binary decision, and risk level.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded yet.")

    raw_features = body.features.model_dump()

    # Engineer features + scale
    X = engineer_features(raw_features)

    # Predict
    proba = model.predict_proba(X)[:, 1][0]
    is_fraud = bool(proba >= BEST_THRESHOLD)

    # Risk level
    if proba < 0.3:
        risk_level = "LOW"
    elif proba < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return PredictResponse(
        is_fraud=is_fraud,
        fraud_probability=round(float(proba), 4),
        risk_level=risk_level,
    )
