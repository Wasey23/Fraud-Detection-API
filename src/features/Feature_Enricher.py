# noinspection PyPackageRequirements
from redis import Redis
# noinspection PyPackageRequirements
from redis import RedisError
import os
import pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEnricher:
    """
    Production-ready feature engineering pipeline.
    Encapsulates time engineering, Redis-based velocity, and target encoding.
    """

    def __init__(self, mapping_path="src/models/saved_models/target_encode_maps.pkl"):
        # 1. Redis Configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_available = False

        try:
            self.redis_client = Redis(host=self.redis_host, port=self.redis_port, db=0)
            self.redis_client.ping()
            self.redis_available = True
            logger.info("FeatureEnricher: Redis connection established.")
        except RedisError:
            logger.warning("FeatureEnricher: Redis unavailable. Fallback to velocity=0.")
            self.redis_available = False

        # 2. Loading Target Encoding Mappings
        try:
            with open(mapping_path, "rb") as f:
                self.target_maps = pickle.load(f)
            logger.info(f"FeatureEnricher: Loaded target encoding maps from {mapping_path}")
        except FileNotFoundError:
            logger.warning("FeatureEnricher: Target encoding maps not found. Encoding will be skipped.")
            self.target_maps = {}

    @staticmethod
    def add_time_features(df):
        df = df.copy()
        df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24
        df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
        return df

    def get_velocity(self, card_id):
        if not self.redis_available:
            return 0
        try:
            val = self.redis_client.get(f"velocity_{card_id}")
            return int(val) if val else 0
        except (RedisError, ValueError) as e:
            logger.warning(f"Could not get velocity for {card_id}: {e}")
            return 0

    def apply_target_encoding(self, df):
        """
        Applies pre-calculated target encoding mappings to categorical features.
        """
        df = df.copy()
        for col, mapping in self.target_maps.items():
            if col in df.columns:
                # Mapping the categories to their pre-calculated values.
                # Filling unknown categories with 0 to prevent crashes.
                df[col] = df[col].map(mapping).fillna(0)
        return df

    def build_all_features(self, df, history=None):
        """
        Orchestrator for all feature engineering steps.
        """
        print(f"DEBUG: History received with length {len(history) if history else 0}")
        # Applying Time Features
        df = self.add_time_features(df)

        # Applying Target Encoding
        df = self.apply_target_encoding(df)

        if history is not None and len(history) > 0:
            # We use a lambda to apply our calculation to every row in the dataframe
            df['card_velocity_10min'] = df.apply(
                lambda row: self._calculate_velocity_from_history(
                    row['TransactionDT'],
                    row['card1'],
                    history,
                    600
                ), axis=1
            )

            df['velocity_count_24h'] = df.apply(
                lambda row: self._calculate_velocity_from_history(
                    row['TransactionDT'],
                    row['card1'],
                    history,
                    86400
                ), axis=1
            )
        else:
            # Fallback if no history is provided (e.g., first transaction ever)
            df['card_velocity_10min'] = 0
            df['velocity_count_24h'] = 0

        return df

    @staticmethod
    def _calculate_velocity_from_history(current_dt, card_id, history, window_seconds):
        """
        Calculates how many times a card appeared in the history
        within the last 600 seconds (10 minutes).
        """
        window_start = current_dt - window_seconds

        # Count transactions in history that match the card_id
        # and fall within the 10-minute time window
        recent_tx = [
            tx for tx in history
            if tx['card1'] == card_id
            and window_start <= tx['TransactionDT'] <= current_dt
        ]
        return len(recent_tx)


