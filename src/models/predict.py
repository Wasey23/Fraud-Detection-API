import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from src.features.build_features import FeatureEnricher
from src.utils.logger import get_logger

logger = get_logger(__name__)

def sweep_thresholds(y_true, y_prob):
    """
    Automates the evaluation of model metrics across multiple thresholds.
    """
    results = []
    # Generate thresholds from 0.1 to 0.9 in increments of 0.1
    thresholds = np.arange(0.1, 1.0, 0.01)

    for t in thresholds:
        # Generate binary predictions based on the current threshold
        y_pred = (y_prob >= t).astype(int)

        # Calculate metrics
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auprc = average_precision_score(y_true, y_pred)
        results.append({
            'threshold': round(t, 2),
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1_score': round(f1, 4),
            'AUPRC': round(auprc, 4)
        })

    return pd.DataFrame(results)

def run_evaluation_and_prediction(csv_path):
    """
    Loads data, engineers features, generates predictions,
    and prints a performance report.
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Pipeline: Feature Engineering
    enricher = FeatureEnricher()
    enriched_df = enricher.build_all_features(df)

    non_numeric_cols = enriched_df.select_dtypes(exclude=['number', 'bool']).columns

    if len(non_numeric_cols) > 0:
        logger.warning(f"Dropping non-numeric columns that were not encoded: {list(non_numeric_cols)}")
        enriched_df = enriched_df.drop(columns=non_numeric_cols)

    # 2. Alignment
    model_path = "src/models/saved_models/xgboost_fraud_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X = enriched_df.select_dtypes(include=['number', 'bool'])
    expected_cols = model.get_booster().feature_names
    # Align columns to match model training
    X = X.reindex(columns=expected_cols, fill_value=0)

    # 3. Prediction
    logger.info("Running predictions...")
    y_prob = model.predict_proba(X.values)[:, 1]

    sweep_df = sweep_thresholds(df['isFraud'], y_prob)
    sweep_df.to_csv("threshold_analysis.csv", index=False)
    print(sweep_df)

    # 4. Evaluation (Assuming 'isFraud' column exists in your data)
    if 'isFraud' in df.columns:
        logger.info("Calculating performance metrics...")
    else:
        logger.warning("No 'isFraud' column found. Skipping evaluation.")

    # 5. Output Sample
    print("\n--- Sample Predictions (First 5) ---")
    print(y_prob[:5])

if __name__ == "__main__":
    # Point this to your training/test data
    run_evaluation_and_prediction("data/raw/train_transaction.csv")

