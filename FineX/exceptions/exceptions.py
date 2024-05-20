class FineXException(Exception):
    """
    A base exception class for the FineX project.
    """

    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code


class InvalidInputError(FineXException):
    """
    An exception for invalid input errors.
    """

    def __init__(self, message):
        super().__init__(message, status_code=400)


class ResourceNotFoundError(FineXException):
    """
    An exception for resource not found errors.
    """

    def __init__(self, message):
        super().__init__(message, status_code=404)


class ResourceConflictError(FineXException):
    """
    An exception for resource conflict errors.
    """

    def __init__(self, message):
        super().__init__(message, status_code=409)


class UnauthorizedError(FineXException):
    """
    An exception for unauthorized errors.
    """

    def __init__(self, message):
        super().__init__(message, status_code=401)


class ForbiddenError(FineXException):
    """
    An exception for forbidden errors.
    """

    def __init__(self, message):
        super().__init__(message, status_code=403)
