# auditing.py
import os
import subprocess

def run_audit_tool(tool: str, input_file: str) -> str:
    process = subprocess.run([tool, input_file], capture_output=True, text=True)
    return process.stdout

def main():
    input_file = "input.txt"
    tools = ["bandit", "safety", "semgrep"]

    for tool in tools:
        output = run_audit_tool(tool, input_file)
        print(f"{tool} output:\n{output}")

if __name__ == "__main__":
    main()
