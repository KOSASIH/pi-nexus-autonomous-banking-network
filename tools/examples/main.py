import code_errors_fixer

code = """
def hello_world():
    print("Hello, world!")

hello_world()
"""

errors = code_errors_fixer.find_syntax_errors(code)

if errors:
    print("Syntax errors found:")
    for error in errors:
        print(error)
else:
    print("No syntax errors found.")
