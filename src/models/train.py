import pandas as pd
import xgboost as xgb
import pickle
import os

def train_and_save_model():
    """Loads engineered data, trains the XGBoost algorithm, and serializes the model."""

    # 1. Load the fully engineered training dataset
    print("Loading engineered data...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/train_engineered.csv'))
    df = pd.read_csv(data_path)
    X = df.select_dtypes(include=['number', 'bool'])
    y = df['isFraud']

    # 2. Instantiate the XGBoost algorithm
    print("Initializing XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    # 3. Train the model
    print("Training the model...")
    model.fit(X, y)

    # 4. Serialize the trained model artifact
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models'))
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'xgboost_fraud_model.pkl')

    print(f"Serializing model artifact to {file_path}...")
    with open(file_path, 'wb') as file:
         pickle.dump(model, file)

    print("Model serialization complete.")

# Execution guard
if __name__ == "__main__":
    train_and_save_model()