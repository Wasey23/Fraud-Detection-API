# src/features/redis_client.py
import redis
import json

class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        # We use a connection pool to manage resource efficiency
        self.pool = redis.ConnectionPool(host=host, port=port, db=db)
        self.client = redis.Redis(connection_pool=self.pool)

    def save_transaction(self, key: str, value: dict):
        # Save transaction with an expiry (e.g., 24 hours) to keep memory clean
        self.client.setex(key, 86400, json.dumps(value))

    def get_history(self, key):
        data = self.client.get(key)
        return json.loads(data) if data else []