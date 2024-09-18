import re

class InputValidator:
    def __init__(self):
        self.patterns = {}

    def add_pattern(self, field, pattern):
        self.patterns[field] = re.compile(pattern)

    def validate(self, data):
        for field, pattern in self.patterns.items():
            if field in data:
                if not pattern.match(data[field]):
                    return False
        return True

# Example usage:
validator = InputValidator()
validator.add_pattern("username", r"^[a-zA-Z0-9_]+$")
validator.add_pattern("email", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

data = {"username": "invalid_username!", "email": "invalid_email"}
if not validator.validate(data):
    print("Invalid input")
