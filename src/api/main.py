from fastapi import FastAPI
import pickle
import os
import sys
import pandas as pd
from src.api.schemas import TransactionRequest

# src directory importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.build_features import FeatureEnricher

# Initializing FastAPI application
app = FastAPI(title='Fraud Detection API')

# Defining the path to serialized data
current_dir = os.path.dirname(__file__)
model_dir = os.path.abspath(os.path.join(current_dir, '../models/saved_models'))
file_path = os.path.join(model_dir, 'target_encode_maps.pkl')

# Loading the serialized data and instantiate the enricher
with open(file_path, 'rb') as file:
    loaded_maps = pickle.load(file)

enricher = FeatureEnricher()
enricher.target_encode_maps = loaded_maps

# health check endpoint
@app.get("/")
def health_check():
    return {"status": "Active", "message": "Fraud Detection API is running and enriched."}

# Core prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    """
    receives a live transaction, enforces validation and processes it
    through the feature enricher
    """

    # Extract the tested data into a python dictionary
    transaction_dict = transaction.model_dump()

    # Transform the dictionary into pandas dataframe
    df = pd.DataFrame([transaction_dict])

    # Executing the orchestration method to make temporal and categorical features
    enriched_df = enricher.build_all_features(df)

    # testing by throwing back at user
    return {
        "status": "success",
        "engineered_features": enriched_df.to_dict(orient='records')[0]
    }