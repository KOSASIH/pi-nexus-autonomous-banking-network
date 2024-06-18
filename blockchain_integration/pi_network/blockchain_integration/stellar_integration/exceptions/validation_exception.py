# validation_exception.py
from stellar_sdk.exceptions import ValidationError as StellarValidationError

class ValidationError(StellarValidationError):
    def __init__(self, message, code, data=None):
        super().__init__(message)
        self.code = code
        self.data = data

    def __str__(self):
        return f"{self.code}: {self.message}"

    def add_validation_rule(self, rule, message):
        self.validation_rules.append((rule, message))

    def validate(self, data):
        for rule, message in self.validation_rules:
            if not rule(data):
                raise self(message, self.code, data)
