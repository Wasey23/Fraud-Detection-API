import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def run_threshold_sweep():
    # 1. Setup paths
    root_dir = r"C:\Users\abdul\Desktop\Fraud Detection API"
    model_path = os.path.join(root_dir, "src", "models", "saved_models", "xgboost_fraud_model.pkl")
    data_path = os.path.join(root_dir, "data", "processed", "train_engineered.csv")

    # 2. Load model and data
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(data_path)

    # 3. Use the temporal tail (the most recent data)
    test_df = df.tail(int(len(df) * 0.2)).copy()
    X_test = test_df.select_dtypes(include=['number', 'bool']).drop(columns=['isFraud'], errors='ignore')
    y_test = test_df['isFraud']

    # Ensure columns match training order
    X_test = X_test.reindex(columns=model.get_booster().feature_names)

    # Get probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # 4. Perform the Sweep
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 50)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"{t:<10.1f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

if __name__ == "__main__":
    run_threshold_sweep()