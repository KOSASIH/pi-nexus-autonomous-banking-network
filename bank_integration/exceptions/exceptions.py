# exceptions.py

class BankIntegrationError(Exception):
    """Base class for all bank integration errors."""

class BankAuthenticationError(BankIntegrationError):
    """Raised when there is an error authenticating with the bank API."""

class BankAPIError(BankIntegrationError):
    """Raised when there is an error with the bank API."""

class BankAccountNotFoundError(BankIntegrationError):
    """Raised when the specified bank account is not found."""

class BankTransactionError(BankIntegrationError):
"""Raised when there is an error making a bank transaction."""

class BankFraudDetectedError(BankIntegrationError):
    """Raised when a bank transaction is flagged as fraudulent."""

class BankDataProcessingError(BankIntegrationError):
    """Raised when there is an error processing bank data."""

class BankNetworkError(BankIntegrationError):
    """Raised when there is a network error while communicating with the bank API."""

class BankEncryptionError(BankIntegrationError):
    """Raised when there is an error encrypting or decrypting bank data."""

class BankDecryptionError(BankIntegrationError):
    """Raised when there is an error decrypting bank data."""

class BankInvalidDataError(BankIntegrationError):
    """Raised when the bank data is invalid or malformed."""

class BankRateLimitExceededError(BankIntegrationError):
    """Raised when the bank API rate limit is exceeded."""

class BankServiceUnavailableError(BankIntegrationError):
    """Raised when the bank API is temporarily unavailable."""

class BankMaintenanceError(BankIntegrationError):
    """Raised when the bank API is down for maintenance."""

class BankUnknownError(BankIntegrationError):
    """Raised when an unknown error occurs while integrating with the bank API."""
