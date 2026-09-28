#!/usr/bin/env python3
"""
scanner.py - One-time hardcoded-secret scanner for the Pi Nexus Banking Network.

Walks a target directory, opens every ``.py`` file, and reports any line that
assigns a hardcoded secret matching a known-bad pattern. This is the detection
half of the Self-Healing Agent (see README.md).

Uses only the Python standard library.

Usage:
    python scanner.py [path]      # path defaults to the current directory
"""

import os
import re
import sys

# The known-bad pattern: an assignment of SECRET / KEY / PASSWORD to one of the
# placeholder values we never want committed to the repository.
SECRET_PATTERN = re.compile(
    r'(SECRET|KEY|PASSWORD)\s*=\s*["\'](super-secret-key|12345|password)["\']'
)

# Directories we never need to scan (keeps the run fast and quiet).
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".mypy_cache"}


def scan_file(path):
    """Return a list of ``(line_number, line_text)`` findings for one file."""
    findings = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, start=1):
                if SECRET_PATTERN.search(line):
                    findings.append((line_number, line.rstrip("\n")))
    except OSError as exc:
        print(f"[!] Could not read {path}: {exc}")
    return findings


def scan_directory(root="."):
    """Walk ``root`` and scan every ``.py`` file.

    Returns a dict of ``{path: [(line_number, line_text), ...]}``.
    """
    results = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in place so os.walk does not descend into them.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                findings = scan_file(full_path)
                if findings:
                    results[full_path] = findings
    return results


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"[*] Scanning .py files under: {os.path.abspath(root)}")

    results = scan_directory(root)

    if not results:
        print("[+] No hardcoded secrets found. Clean!")
        return 0

    total = sum(len(found) for found in results.values())
    print(f"[!] Found {total} hardcoded secret(s) in {len(results)} file(s):\n")
    for path, findings in results.items():
        for line_number, line_text in findings:
            print(f"  {path}:{line_number}: {line_text.strip()}")

    print("\n[i] Run healer.py to auto-fix these findings.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
