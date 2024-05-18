import ast


def find_syntax_errors(code: str) -> list:
    """
    Finds syntax errors in a string of Python code.

    Args:
    - code (str): The string of Python code to parse.

    Returns:
    - A list of syntax errors found in the code.
    """

    try:
        # Parse the code using the ast module
        tree = ast.parse(code)
    except SyntaxError as e:
        # Return the syntax error as a list
        return [str(e)]
    else:
        # If the code is syntactically correct, return an empty list
        return []
