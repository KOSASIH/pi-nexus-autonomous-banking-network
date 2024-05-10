def reverse_string(s: str) -> str:
    """
    Returns the reverse of the given string.
    """
    return s[::-1]


def truncate_string(s: str, length: int) -> str:
    """
    Returns the given string truncated to the given length.
    """
    if len(s) > length:
        return s[:length] + "..."
    else:
        return s


def count_words(s: str) -> int:
    """
    Returns the number of words in the given string.
    """
    return len(s.split())


def count_vowels(s: str) -> int:
    """
    Returns the number of vowels in the given string.
    """
    vowels = "aeiou"
    return sum(1 for c in s.lower() if c in vowels)
