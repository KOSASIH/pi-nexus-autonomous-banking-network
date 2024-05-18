import glob
import os


def analyze_codebase():
    languages = set()
    tools = set()

    # Analyze the codebase to identify the programming languages used and the development tools required
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                language = get_language(file)
                tools.update(get_tools(file))
                languages.add(language)

    return languages, tools


def get_language(file):
    # Determine the programming language based on the file extension
    if file.endswith(".py"):
        return "Python"


def get_tools(file):
    # Determine the development tools required based on the file content
    tools = set()
    if "numpy" in open(file).read():
        tools.add("numpy")
    if "pandas" in open(file).read():
        tools.add("pandas")
    return tools


if __name__ == "__main__":
    languages, tools = analyze_codebase()
    print(f"Programming languages used: {languages}")
    print(f"Development tools required: {tools}")
