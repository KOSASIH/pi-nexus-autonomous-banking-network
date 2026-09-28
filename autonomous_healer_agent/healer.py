#!/usr/bin/env python3
"""
healer.py - One-time auto-healer for hardcoded secrets.

Uses the same detection rule as ``scanner.py`` to find hardcoded secrets, then
rewrites each one to read from the environment instead:

    JWT_SECRET_KEY = <hardcoded value>   ->   JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

The healer also guarantees the file imports ``os``. This is the repair half of
the Self-Healing Agent (see README.md).

Uses only the Python standard library.

Usage:
    python healer.py [path]      # path defaults to the current directory
"""

import os
import re
import sys

from scanner import SECRET_PATTERN, scan_directory

# What we replace the hardcoded value with.
ENV_LOOKUP = 'os.getenv("JWT_SECRET_KEY")'

# A line that already reads the value from the environment is left untouched.
ENV_LINE_PATTERN = re.compile(r'os\.getenv\(\s*["\']JWT_SECRET_KEY["\']\s*\)')

# Matches only the quoted secret literal, e.g. "super-secret-key" or 'password'.
SECRET_LITERAL_PATTERN = re.compile(r'["\'](super-secret-key|12345|password)["\']')


def ensure_os_import(text):
    """Return ``text`` with an ``import os`` line guaranteed to be present."""
    if re.search(r"^\s*import\s+os\b", text, re.MULTILINE):
        return text
    if re.search(r"^\s*from\s+os\s+import\b", text, re.MULTILINE):
        return text
    return "import os\n" + text


def heal_file(path):
    """Rewrite hardcoded secrets in a single file.

    Returns a list of human-readable descriptions of the fixes applied.
    """
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    fixes = []
    changed = False
    for index, line in enumerate(lines):
        if not SECRET_PATTERN.search(line):
            continue
        if ENV_LINE_PATTERN.search(line):
            continue

        new_line = SECRET_LITERAL_PATTERN.sub(ENV_LOOKUP, line)
        if new_line != line:
            fixes.append(f"  line {index + 1}: {line.strip()}  ->  {new_line.strip()}")
            lines[index] = new_line
            changed = True

    if changed:
        healed_text = ensure_os_import("".join(lines))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(healed_text)

    return fixes


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"[*] Healing hardcoded secrets under: {os.path.abspath(root)}")

    results = scan_directory(root)

    if not results:
        print("[+] Nothing to heal. No hardcoded secrets found.")
        return 0

    total_fixed = 0
    for path in results:
        fixes = heal_file(path)
        if fixes:
            print(f"[+] Fixed {path}:")
            for fix in fixes:
                print(fix)
            total_fixed += len(fixes)

    if total_fixed == 0:
        print("[+] No changes were needed.")
    else:
        print(f"\n[+] Done. Applied {total_fixed} fix(es).")
        print("[i] Remember to set the JWT_SECRET_KEY environment variable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
