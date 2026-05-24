import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI
from src.api.schemas import TransactionRequest
from src.features.build_features import FeatureEnricher
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Load artifacts (The "Brain" and the "Mapper")
# These are loaded once at startup to ensure high performance during requests
MODEL_PATH = "src/models/saved_models/xgboost_fraud_model.pkl"
ENCODER_PATH = "src/models/saved_models/target_encode_maps.pkl"

logger.info("Loading model artifacts...")
with open(MODEL_PATH, "rb") as f:
    fraud_model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    target_encode_maps = pickle.load(f)

# Instantiate the FeatureEnricher
enricher = FeatureEnricher()
logger.info("Artifacts loaded and FeatureEnricher initialized.")

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    """
    Receives a transaction, processes features, and returns a fraud risk score.
    """
    logger.info(f"Received transaction for Card ID: {transaction.card1}")

    # 1. Pipeline: Feature Engineering
    df = pd.DataFrame([transaction.model_dump()])
    enriched_df = enricher.build_all_features(df)

    # 2. Pipeline: Data Alignment
    # Filter for numeric features and reindex to match training column order
    X_live = enriched_df.select_dtypes(include=['number', 'bool'])
    expected_cols = fraud_model.get_booster().feature_names
    X_live = X_live.reindex(columns=expected_cols, fill_value=0)

    # 3. Pipeline: Inference
    fraud_probability = float(fraud_model.predict_proba(X_live.values)[0][1])

    # 4. Pipeline: Business Logic
    risk_level = "High" if fraud_probability > 0.5 else "Low"

    # Log outcome for audit trail
    if risk_level == "High":
        logger.warning(f"BLOCKED: Fraud Score {fraud_probability:.4f} for Card {transaction.card1}")
    else:
        logger.info(f"APPROVED: Fraud Score {fraud_probability:.4f} for Card {transaction.card1}")

    return {
        "status": "success",
        "fraud_probability": round(fraud_probability, 4),
        "risk_level": risk_level
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)