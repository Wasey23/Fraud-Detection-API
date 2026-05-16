import pandas as pd

class FeatureEnricher:
    def __init__(self):
        self.target_encode_maps = {}

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
            new_col_name = f"{col}_target_enc"

            global_mean = mapping.get('global_mean', 0)

            new_features[new_col_name] = df[col].map(mapping).fillna(global_mean)

        return pd.DataFrame(new_features, index=df.index)

    def build_all_features(self, df):
        '''Orchestrates feature engineering and safely concatenates to prevent fragmentation'''
        # Generating one-time features
        time_features_df = self.engineer_time_features(df)

        # Generate encoded categorical features
        encoded_features_df = self.apply_target_encoding(df)

        # Concatenate all dataframes simulatneously along the column axis
        final_df = pd.concat([df, time_features_df, encoded_features_df], axis=1)

        return final_df
