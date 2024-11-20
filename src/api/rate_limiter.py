import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, rate_limit: int, time_window: int):
        """
        Initialize the RateLimiter.

        :param rate_limit: Maximum number of requests allowed in the time window.
        :param time_window: Time window in seconds for the rate limit.
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.requests = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        """
        Check if a request is allowed for the given user.

        :param user_id: Unique identifier for the user.
        :return: True if the request is allowed, False otherwise.
        """
        current_time = time.time()
        # Clean up old requests
        self.requests[user_id] = [timestamp for timestamp in self.requests[user_id] if timestamp > current_time - self.time_window]

        if len(self.requests[user_id]) < self.rate_limit:
            # Allow the request
            self.requests[user_id].append(current_time)
            return True
        else:
            # Deny the request
            return False

# Example usage
if __name__ == "__main__":
    rate_limiter = RateLimiter(rate_limit=5, time_window=10)  # 5 requests per 10 seconds

    user_id = "user123"

    for i in range(10):
        if rate_limiter.is_allowed(user_id):
            print(f"Request {i + 1}: Allowed")
        else:
            print(f"Request {i + 1}: Denied")
        time.sleep(1)  # Simulate a delay between requests
