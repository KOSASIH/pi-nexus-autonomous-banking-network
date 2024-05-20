from flask import request, Response
from functools import wraps

def error_handler(func):
    """
    An error handling middleware for the FineX project.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return Response(
                response=str(e),
                status=500,
                mimetype='text/plain'
            )
    return wrapper

def rate_limiter(max_requests=1, duration=1):
    """
    A rate limiter middleware for the FineX project.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ip = request.remote_addr
            if ip in rate_limit_cache:
                if rate_limit_cache[ip] >= max_requests:
                    return Response(
                        response='Too many requests',
                        status=429,
                        mimetype='text/plain'
                    )
                else:
                    rate_limit_cache[ip] += 1
            else:
                rate_limit_cache[ip] = 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cache(duration=300):
    """
    A caching middleware for the FineX project.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f'{func.__name__}:{request.path}:{request.query_string}'
            cached_response = cache.get(cache_key)
            if cached_response:
                return cached_response
            else:
                response = func(*args, **kwargs)
                cache.set(cache_key, response, timeout=duration)
                return response
        return wrapper
    return decorator
