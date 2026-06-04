import redis
import json
import os
import logging

logger = logging.getLogger(__name__)

class RedisClient:
    # Class Attributes: These exist at the class level and are initialized
    # immediately when the module is imported, removing the need for __init__ instantiation.
    _host = os.getenv("REDIS_HOST", "localhost")
    _port = int(os.getenv("REDIS_PORT", 6379))

    # We use a shared connection pool at the class level to manage resource efficiency
    _pool = redis.ConnectionPool(host=_host, port=_port, db=0, socket_timeout=2)
    _client = redis.Redis(connection_pool=_pool)

    @classmethod
    def save_transaction(cls, key: str, value: dict):
        """
        Caches a serialized transaction payload with a 24-hour expiration window.
        """
        try:
            cls._client.rpush(key, json.dumps(value))
            cls._client.ltrim(key, -250, -1)
            cls._client.expire(key, 86400)
        except Exception as e:
            logger.warning(f"Redis write error for key {key}: {e}")

    @classmethod
    def get_history(cls, key: str):
        """
        Retrieves historical transaction records bound to a specific transaction key.
        """
        try:
            raw_data = cls._client.lrange(key, 0, -1)
            return [json.loads(d) for d in raw_data] if raw_data else []
        except Exception as e:
            logger.warning(f"Redis read error for key {key}: {e}")
            return []