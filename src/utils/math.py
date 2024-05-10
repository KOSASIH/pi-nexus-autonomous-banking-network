import math

def factorial(n: int) -> int:
    """
    Returns the factorial of the given number.
    """
    return math.factorial(n)

def is_prime(n: int) -> bool:
    """
    Returns True if the given number is prime, False otherwise.
    """
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def gcd(a: int, b: int) -> int:
    """
    Returns the greatest common divisor of the given numbers.
    """
    return math.gcd(a, b)

def lcm(a: int, b: int) -> int:
    """
    Returns the least common multiple of the given numbers.
    """
    return a * b // math.gcd(a, b)
