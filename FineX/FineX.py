# FineX.py


class FineXException(Exception):
    """Base exception class for FineX"""

    pass


class FineXError(FineXException):
    """Error exception class for FineX"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FineXWarning(FineXException):
    """Warning exception class for FineX"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FineX:
    """FineX class"""

    def __init__(self, name):
        self.name = name

    def greet(self):
        """Print a greeting message"""
        print(f"Hello, {self.name}!")

    def add_numbers(self, a, b):
        """Add two numbers and return the result"""
        return a + b


def main():
    """Main function"""
    fine_x = FineX("John")
    fine_x.greet()
    result = fine_x.add_numbers(2, 3)
    print(f"The result is: {result}")


if __name__ == "__main__":
    main()
