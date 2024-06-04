class PiWalletException(Exception):
    pass

class PiWalletInsufficientBalanceException(PiWalletException):
    pass

class PiWalletInvalidPrivateKeyException(PiWalletException):
    pass

class PiWalletInvalidPublicKeyException(PiWalletException):
    pass

class PiWalletInvalidSignatureException(PiWalletException):
    pass

class PiWalletTransactionNotFoundException(PiWalletException):
    pass
