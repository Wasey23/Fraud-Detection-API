import pandas as pd
from src.features.redis_client import get_redis_client

class FeatureEnricher:
    def __init__(self):
        self.redis = get_redis_client()
        self.target_encode_maps = {}

    def get_card_velocity(self, card_id, window_seconds=600):
        """
        Uses Redis to track transaction count per card over a sliding window.
        window_seconds defaults to 600 (10 minutes).
        """
        key = f"velocity:card:{card_id}"

        # INCR: Atomic operation to increment the counter for this card
        count = self.redis.incr(key)

        # EXPIRE: Set the TTL (Time-To-Live) for this key
        # We only set it if it is the first transaction (count == 1)
        if count == 1:
            self.redis.expire(key, window_seconds)

        return count

    def engineer_time_features(self, df):
        """making cyclical time features"""
        new_features = {}

        new_features['hour_of_day'] = (df['TransactionDT'] // 3600) % 24
        new_features['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7

        return pd.DataFrame(new_features, index=df.index)

    def apply_target_encoding(self, df):
        """Applies pre-calculated fraud probabilities to categorical features"""
        new_features = {}
        for col, mapping in self.target_encode_maps.items():
            if col in df.columns:
                new_col_name = f"{col}_target_enc"
                global_mean = mapping.get('global_mean', 0)

                new_features[new_col_name] = df[col].map(mapping).fillna(global_mean)

        return pd.DataFrame(new_features, index=df.index)

    def build_all_features(self, df):
        '''Orchestrates feature engineering and safely concatenates to prevent fragmentation'''
        # 1. Initialize: Work on a copy to preserve the integrity of the original input
        processed_df = df.copy()

        # 2. Generate components
        time_features_df = self.engineer_time_features(df)
        encoded_features_df = self.apply_target_encoding(df)

        # 3. Combine: Concatenate all features into one unified object
        # We use a list of dataframes to combine them simultaneously
        final_df = pd.concat([processed_df, time_features_df, encoded_features_df], axis=1)

        # 4. Integrate: Add velocity features directly to the unified final_df
        # This ensures your final output contains every single feature we engineered
        final_df['card1_velocity_10min'] = df['card1'].apply(
        lambda x: self.get_card_velocity(x)
        )

        # 5. Exit: The single, authoritative return point
        return final_df
