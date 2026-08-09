import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import your custom feature engineering class from your pipeline
from src.features.Feature_Enricher import FeatureEnricher

# (Optional but recommended) Bring over your logger if your data-loading logic uses it
from src.utils.logger import get_logger
logger = get_logger(__name__)

logger.info("Loading data...")
df_trans = pd.read_csv("data/raw/train_transaction.csv")
df_id = pd.read_csv("data/raw/train_identity.csv")

# 2. Merge Data (Left Join)
logger.info("Merging transaction and identity data...")
df = df_trans.merge(df_id, on="TransactionID", how="left")

enricher = FeatureEnricher()
logger.info("Engineering features...")
enriched_df = enricher.build_all_features(df)

X = enriched_df.select_dtypes(include=['number', 'bool'])
X = X.drop(columns=['isFraud'], errors='ignore')
y = df['isFraud']
# Split the data EXACTLY like train.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prevent a harmless but annoying Pandas warning
X_test = X_test.copy()

# 2. Apply Target Encoding (THE FIX)
print("Applying target encodings...")
with open("src/models/saved_models/target_encode_maps.pkl", "rb") as f:
    target_encode_maps = pickle.load(f)

# Loop through our saved dictionaries and convert strings to numbers
for col, mapping in target_encode_maps.items():
    if col in X_test.columns:
        # Replace the text with the mapped number. If it sees a brand-new category, give it a 0.
        X_test[col] = X_test[col].map(mapping).fillna(0)

# Convert any remaining leftover string columns to numeric just in case
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# 3. Deserialize (Thaw out) the Models
print("Loading models into memory...")
with open("src/models/saved_models/xgboost_fraud_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("src/models/saved_models/lightgbm_fraud_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

print("Aligning features...")
X_test_xgb = X_test[xgb_model.get_booster().feature_names]
X_test_lgb = X_test[lgb_model.feature_name_]

# 4. Generate Predictions (Inference)
print("Generating XGBoost Report...")
xgb_predictions = xgb_model.predict(X_test_xgb)
print(classification_report(y_test, xgb_predictions))

print("Generating LightGBM Report...")
lgb_predictions = lgb_model.predict(X_test_lgb)
print(classification_report(y_test, lgb_predictions))