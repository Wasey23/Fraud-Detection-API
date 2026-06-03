from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from contextlib import asynccontextmanager
from src.api.schemas import TransactionInput, FraudPredictionResponse
from src.features.build_features import FeatureEnricher
from src.features.redis_client import RedisClient
from src.utils.logger import get_logger
from typing import Any, Optional

model: Any = None
enricher: Optional[Any] = None
EXPECTED_COLS: Optional[list] = None

# Initializing logger
logger = get_logger(__name__)

# Optimized business threshold
CUSTOMER_PRECISION_THRESHOLD = 0.85

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Modern FastAPI lifespan context manager.
    Replaces the deprecated @app.on_event("startup").
    Loads the heavy ML models into RAM before the server accepts requests.
    """
    global model, enricher, EXPECTED_COLS
    logger.info("Initializing API and loading machine learning artifacts...")

    try:
        # 1. Loading the XGBoost Brain
        with open("src/models/saved_models/xgboost_fraud_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Capturing the order of columns the model expects
        EXPECTED_COLS = model.get_booster().feature_names

        # 2. Loading the Feature Pipeline
        enricher = FeatureEnricher()

        logger.info("All artifacts loaded successfully. API is ready to serve.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise RuntimeError("Could not boot API. Missing ML artifacts.")

    yield # This yields control back to FastAPI to run the server

    # Any cleanup code for when the server shuts down goes here
    logger.info("Shutting down API and cleaning up resources.")

# Initializing FastAPI application with the lifespan manager
app = FastAPI(
    title="Real-Time Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    """
    The primary endpoint. Accepts a transaction, processes it, and returns a decision.
    """
    try:
        # 1. Defensively Isolate Infrastructure Reads
        # If the database goes offline, capture the exception and fall back to an empty history
        # so the entire payment processing API doesn't crash.
        try:
            history = RedisClient.get_history(f"card:{transaction.card1}")
        except Exception as net_err:
            logger.warning(f"Redis infrastructure lookup failed: {net_err}. Falling back to empty historical window.")
            history = []

        # 2. Tabular Vectorization
        df = pd.DataFrame([transaction.model_dump()])

        # 3. Pipeline: Feature Engineering
        enriched_features = enricher.build_all_features(df, history)

        # 4. Data Sanitization & Alignment
        X = enriched_features.select_dtypes(include=['number', 'bool'])
        X = X.reindex(columns=EXPECTED_COLS, fill_value=0)

        # 5. Inference Processing (Preserving Feature Names)
        probabilities = model.predict_proba(X)
        probability = float(probabilities[0, 1])

        # 6. Defensively Isolating Infrastructure Writes
        try:
            RedisClient.save_transaction(
                key=f"card:{transaction.card1}",
                value=transaction.model_dump()
            )
        except Exception as net_err:
            logger.warning(f"Redis infrastructure write failed: {net_err}. Transaction processing un-cached.")

        # 7. Applying the Business Threshold
        is_fraud = int(probability >= CUSTOMER_PRECISION_THRESHOLD)
        status = "BLOCKED" if is_fraud else "APPROVED"

        # 8. Return the strictly formatted response contract
        return FraudPredictionResponse(
            transaction_id=transaction.TransactionID,
            is_fraud=is_fraud,
            fraud_probability=round(probability, 4),
            decision_threshold=CUSTOMER_PRECISION_THRESHOLD,
            status=status
        )

    except Exception as e:
        logger.error(f"Catastrophic inference pipeline crash: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during real-time prediction sequence."
        )