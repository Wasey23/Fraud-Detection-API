import redis
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
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=0)
            self.redis_client.ping()
            self.redis_available = True
            logger.info("FeatureEnricher: Redis connection established.")
        except redis.ConnectionError:
            logger.warning("FeatureEnricher: Redis unavailable. Fallback to velocity=0.")
            self.redis_available = False

        # 2. Load Target Encoding Mappings
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
        except (redis.exceptions.RedisError, ValueError) as e:
            logger.warning(f"Could not get velocity for {card_id}: {e}")
            return 0

    def apply_target_encoding(self, df):
        """
        Applies pre-calculated target encoding mappings to categorical features.
        """
        df = df.copy()
        for col, mapping in self.target_maps.items():
            if col in df.columns:
                # Map the categories to their pre-calculated values.
                # Fill unknown categories with 0 to prevent crashes.
                df[col] = df[col].map(mapping).fillna(0)
        return df

    def build_all_features(self, df):
        """
        Orchestrator for all feature engineering steps.
        """
        # 1. Apply Time Features
        df = self.add_time_features(df)

        # 2. Apply Redis Velocity Features
        df['card1_velocity_10min'] = df['card1'].apply(self.get_velocity)

        # 3. Apply Target Encoding
        df = self.apply_target_encoding(df)

        return df
