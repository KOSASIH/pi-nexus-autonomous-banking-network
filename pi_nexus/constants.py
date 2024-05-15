# Improved code with type hints and constant naming conventions
from typing import Final


class Constants:
    API_VERSION: Final[str] = "v1"
    BANK_API_URL: Final[str] = "https://api.example.com/bank"
    TRANSACTION_LIMIT: Final[int] = 1000
    # ...
