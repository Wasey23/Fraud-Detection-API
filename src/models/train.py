import pandas as pd
import os
import pickle
import xgboost as xgb
from src.features.build_features import FeatureEnricher
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_model():
    # 1. Load Data
    print("Loading data...")
    # noinspection PyTypeChecker
    df_trans = pd.read_csv("data/raw/train_transaction.csv")
    # noinspection PyTypeChecker
    df_id = pd.read_csv("data/raw/train_identity.csv")

    # 2. Merge Data (Left Join)
    # We join on 'TransactionID' so every transaction remains in the set
    print("Merging transaction and identity data...")
    df = df_trans.merge(df_id, on="TransactionID", how="left")

    # 3. Instantiate FeatureEnricher
    # Now the enricher will see the full merged DataFrame
    enricher = FeatureEnricher()
    print("Engineering features...")
    enriched_df = enricher.build_all_features(df)

    non_numeric_cols = enriched_df.select_dtypes(exclude=['number', 'bool']).columns

    if len(non_numeric_cols) > 0:
        logger.warning(f"Dropping non-numeric columns that were not encoded: {list(non_numeric_cols)}")
        enriched_df = enriched_df.drop(columns=non_numeric_cols)
    # 4. Prepare X and y
    # Selecting features (numeric/bool) to ensure model compatibility
    X = enriched_df.select_dtypes(include=['number', 'bool'])
    X = X.drop(columns=['isFraud'], errors='ignore')
    y = df['isFraud']

    print("Training XGBoost model with class balancing...")

    ratio = float(y.value_counts()[0]/y.value_counts()[0])
    # 5. Train Model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(n_estimators=100,
                              max_depth=6,
                              scale_pos_weight=ratio)
    model.fit(X, y)

    # Get feature importances
    importance = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("\n--- Top 10 Features Used by the Model ---")
    print(feature_importance_df.head(10))

    # 6. Save Artifacts
    save_path = "src/models/saved_models/xgboost_fraud_model.pkl"
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Using atomic save pattern
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(model, f)
    os.replace(tmp_path, save_path)

    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()