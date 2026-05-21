import pandas as pd
from src.features.build_features import FeatureEnricher

def test_feature_enricher_output_shape():
    """Validates that the enricher adds expected columns to the DataFrame."""

    # 1. Arrange: Create dummy transaction data
    enricher = FeatureEnricher()
    df = pd.DataFrame([{
        "TransactionDT": 86400,
        "card1": 12500,
        "TransactionAmt": 250.75,
        "P_emaildomain": "gmail.com"
    }])

    # 2. Act: Run the transformation
    enriched_df = enricher.build_all_features(df)

    # 3. Assert: Check the result
    # We expect the enriched dataframe to have more columns than the input
    assert enriched_df.shape[1] > df.shape[1], "Enricher failed to add features."
    assert 'TransactionDT' in enriched_df.columns, "Core column missing."