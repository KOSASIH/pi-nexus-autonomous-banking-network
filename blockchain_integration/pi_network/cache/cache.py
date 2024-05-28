# cache.py

import redis
from functools import wraps

class Cache:
    def __init__(self, host, port, db):
        self.redis_client = redis.Redis(host=host, port=port, db=db)

    def get(self, key):
        return self.redis_client.get(key)

    def set(self, key, value, expire=3600):
        self.redis_client.set(key, value, expire)

    def delete(self, key):
        self.redis_client.delete(key)

def cache_decorator(ttl=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__module__}:{func.__name__}:{args}:{kwargs}"
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

cache = Cache(host="localhost", port=6379, db=0)

# usage
@cache_decorator(ttl=3600)
def get_balance(address):
    # API call to fetch balance
    return {"balance": 100}

result = get_balance("my_address")
print(result)  # returns cached value if available
