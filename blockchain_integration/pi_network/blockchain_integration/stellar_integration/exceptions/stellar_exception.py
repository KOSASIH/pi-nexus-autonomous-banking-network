class StellarException(Exception):
    pass

class StellarNetworkError(StellarException):
    pass

class StellarInvalidTransactionError(StellarException):
    pass

class StellarInsufficientBalanceError(StellarException):
    pass

class StellarAccountNotFoundError(StellarException):
    pass

class StellarTransactionNotFoundError(StellarException):
    pass
