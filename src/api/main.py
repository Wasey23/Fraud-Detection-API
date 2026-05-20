from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import sys

# Ensure the src directory is in the system path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.build_features import FeatureEnricher
from src.api.schemas import TransactionRequest

# 1. Initialize the FastAPI application
app = FastAPI(title="Fraud Detection API")

# 2. Define the path to your serialized artifacts
current_dir = os.path.dirname(__file__)
model_dir = os.path.abspath(os.path.join(current_dir, '../models/saved_models'))

maps_path = os.path.join(model_dir, 'target_encode_maps.pkl')
model_path = os.path.join(model_dir, 'xgboost_fraud_model.pkl')

# 3. Load the artifacts into active memory
print("Loading historical data mappings...")
with open(maps_path, 'rb') as file:
    loaded_maps = pickle.load(file)

print("Loading XGBoost model artifact...")
with open(model_path, 'rb') as file:
    fraud_model = pickle.load(file)

# Instantiate the feature engineering blueprint
enricher = FeatureEnricher()
enricher.target_encode_maps = loaded_maps

# 4. Define a basic health-check endpoint
@app.get("/")
def health_check():
    return {"status": "Active", "message": "Fraud Detection API is running and enriched."}

# 5. Define the core prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    """
    Receives a live transaction, enforces validation, engineers features,
    and returns a calculated fraud probability score.
    """

    # Extract data and convert to a Pandas DataFrame
    transaction_dict = transaction.model_dump()
    df = pd.DataFrame([transaction_dict])

    # Execute the orchestration method to engineer features
    enriched_df = enricher.build_all_features(df)

    # Apply the same strict mathematical filter we used during training
    X_live = enriched_df.select_dtypes(include=['number', 'bool'])

    expected_columns = fraud_model.get_booster().feature_names
    X_live = X_live.reindex(columns=expected_columns)

    raw_matrix = X_live.values
    # Execute real-time inference
    # .predict_proba() returns a nested array: [[probability_of_0, probability_of_1]]
    fraud_probability = float(fraud_model.predict_proba(raw_matrix)[0][1])

    # Define a basic business logic threshold
    risk_level = "High" if fraud_probability > 0.5 else "Low"

    return {
        "status": "success",
        "fraud_probability": round(fraud_probability, 4),
        "risk_level": risk_level
    }