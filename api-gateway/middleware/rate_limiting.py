from flask import g
from time import time

def rate_limit(func):
    def wrapper(*args, **kwargs):
        rate_limit_interval = app.config['API_GATEWAY_RATE_LIMITING_INTERVAL']
        rate_limit_limit = app.config['API_GATEWAY_RATE_LIMITING_LIMIT']

        rate_limiting_key = f'rate_limiting_{request.remote_addr}'
        current_count = g.cache.get(rate_limiting_key) or 0

        if current_count >= rate_limit_limit:
            return jsonify({'message': 'Rate limit exceeded'}), 429

        g.cache.set(rate_limiting_key, current_count + 1, timeout=rate_limit_interval)

        return func(*args, **kwargs)

    return wrapper
