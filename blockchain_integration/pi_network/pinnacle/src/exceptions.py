class PinnacleException(Exception):
    pass

class ConfigException(PinnacleException):
    pass

class BlockchainException(PinnacleException):
    pass

class APIException(PinnacleException):
    pass

class GasPriceException(PinnacleException):
    pass
