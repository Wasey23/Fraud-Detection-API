import pandas as pd
import numpy as np
import os
import pickle
import gc
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, classification_report
from src.features.Feature_Enricher import FeatureEnricher
from src.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_offline_velocity(df):
    """
    Simulates the live Redis database by calculating historical
    velocity features directly from the static training dataset.
    """
    print("Calculating historical velocity features...")
    df = df.copy()
    # 1. Create a datetime column for the rolling window
    df['temp_time'] = pd.to_datetime(df['TransactionDT'], unit='s')

    # 2. Sorting explicitly by card first, then by time.
    # This guarantees the DataFrame row order perfectly matches the GroupBy output order.
    df = df.sort_values(by=['card1', 'temp_time']).reset_index(drop=True)

    # 3. Temporarily set the index to time so rolling() works flawlessly
    df_indexed = df.set_index('temp_time')

    # 4. Calculate 24-hour velocity
    roll_24h = df_indexed.groupby('card1')['TransactionID'].rolling('24h').count()
    # BYPASS ALIGNMENT: Since the DF and Series are perfectly ordered,
    # we just extract the raw numpy array (.values) and paste it in.
    df['velocity_count_24h'] = roll_24h.values

    # 5. Calculate 10-minute velocity
    print("Calculating 10-minute velocity...")
    roll_10m = df_indexed.groupby('card1')['TransactionID'].rolling('10min').count()
    df['card_velocity_10min'] = roll_10m.values - 1

    # 6. Cleanup temporary columns
    df.drop(columns=['temp_time'], inplace=True)

    # 7. Sort back to pure chronological order to match how data naturally arrived
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    return df

def sweep_thresholds(y_true, y_prob):
    """
    Automates the evaluation of model metrics across multiple thresholds.
    """
    results = []
    thresholds = np.arange(0.1, 1.0, 0.1)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auprc = average_precision_score(y_true, y_prob) # AUPRC uses probabilities, not binary predictions

        results.append({
            'threshold': round(t, 2),
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1_score': round(f1, 4),
            'auprc': round(auprc, 4)
        })
    return pd.DataFrame(results)

def drop_dead_weight(model, X_train):
    """
    Identifies features with 0.0 importance and drops them
    to compress the dataset and speed up API inference.
    """
    print("Scanning for dead weight features...")

    # Get the importance scores from the trained model
    importances = model.feature_importances_

    # Zip them together with the column names
    feature_scores = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    })

    # Find all columns that scored exactly 0.0
    dead_features = feature_scores[feature_scores['importance'] == 0.0]['feature'].tolist()

    print(f"Dropping {len(dead_features)} useless features!")

    # Drop them from the dataset
    X_train_lean = X_train.drop(columns=dead_features)

    return X_train_lean, dead_features

def train_model():
    # 1. Load Data
    logger.info("Loading data...")
    df_trans = pd.read_csv("data/raw/train_transaction.csv")
    df_id = pd.read_csv("data/raw/train_identity.csv")

    # 2. Merge Data (Left Join)
    logger.info("Merging transaction and identity data...")
    df = df_trans.merge(df_id, on="TransactionID", how="left")

    print("Simulating Redis...")
    df = calculate_offline_velocity(df)

    # 3. Instantiate FeatureEnricher
    enricher = FeatureEnricher()
    logger.info("Engineering features...")
    enriched_df = enricher.build_all_features(df)

    non_numeric_cols = enriched_df.select_dtypes(exclude=['number', 'bool']).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Dropping non-numeric columns that were not encoded: {list(non_numeric_cols)}")
        enriched_df = enriched_df.drop(columns=non_numeric_cols)

    # 4. Prepare X and y
    X = enriched_df.select_dtypes(include=['number', 'bool'])
    X = X.drop(columns=['isFraud'], errors='ignore')
    y = df['isFraud']

    logger.info("Executing Garbage collection to free system RAM")
    del df_trans, df_id, df, enriched_df
    gc.collect()

    # Train/Test Split ---
    logger.info("Splitting data into training and evaluation sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale Pos Weight Calculation
    logger.info("Calculating class imbalance ratio...")
    ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    logger.info(f"Scale positive weight set to: {ratio:.2f}")

    # 5. Train XGBoost Model
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        scale_pos_weight=ratio,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Train LightGBM Model
    logger.info("Training LightGBM model...")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=ratio,
        random_state=42,
        num_threads=1
    )
    lgb_model.fit(X_train, y_train)

    # 6. Evaluate Model on Holdout Set
    logger.info("Running predictions on the holdout test set...")
    y_prob = model.predict_proba(X_test)[:, 1]

    logger.info("Evaluating thresholds...")
    sweep_df = sweep_thresholds(y_test, y_prob)
    print("\n--- Model Performance Metrics ---")
    print(sweep_df.to_string(index=False))

    # 7. Feature Importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    print("\n--- Top 10 Features Used by the Model ---")
    print(feature_importance_df.head(30).to_string(index=False))

    X_train_lean, dropped_list = drop_dead_weight(model, X_train)
    print("List of Features Dropped")
    print(dropped_list)

    # 8. Quality Gate
    final_auprc = average_precision_score(y_test, y_prob)
    BUSINESS_THRESHOLD = 0.50
    y_pred_default = (y_prob >= BUSINESS_THRESHOLD).astype(int)
    final_recall = recall_score(y_test, y_pred_default, zero_division=0)

    MINIMUM_AUPRC_REQUIRED = 0.65
    MINIMUM_RECALL_REQUIRED = 0.80

    if final_auprc >= MINIMUM_AUPRC_REQUIRED and final_recall >= MINIMUM_RECALL_REQUIRED:
        save_path = "src/models/saved_models/xgboost_fraud_model.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        temp_path = save_path + ".tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(model, f)
        os.replace(temp_path, save_path)

        print(f"\nClassification Report")
        print(classification_report(y_test, y_pred_default, zero_division=0))
        logger.info(f"Model successfully saved to {save_path}")
    else:
        print(f"\nClassification Report")
        print(classification_report(y_test, y_pred_default, zero_division=0))
        logger.warning(f"FAILURE: Model rejected. AUPRC ({final_auprc:.4f}) or Recall ({final_recall:.4f}) failed to meet requirements.")

    lgb_save_path = "src/models/saved_models/lightgbm_fraud_model.pkl"
    with open(lgb_save_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"LGBM Model successfully saved to {lgb_save_path}")

if __name__ == '__main__':
    train_model()
