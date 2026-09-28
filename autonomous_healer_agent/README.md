# Autonomous Healer Agent

A lightweight **Self-Healing Agent** for the **Pi Nexus Autonomous Banking Network**.

The agent watches for a specific class of security bug — **hardcoded secrets** —
and repairs it automatically. It is intentionally small, dependency-free, and
safe to run: it performs a **single one-time scan** (there is no 24/7 daemon
loop), and it only touches `.py` files that match a known-bad pattern.

## Why this exists

Banking infrastructure must never ship credentials inside its source code.
A leaked `JWT_SECRET_KEY`, database password, or API key committed to a public
repository can compromise token signing, session integrity, and customer data.
This agent gives the Pi Nexus codebase a self-healing safety net: it detects the
mistake and rewrites the code to pull the value from the environment instead.

## The problem it fixes

Consider this vulnerable line, which was found in
`blockchain_integration/pi_network/PiEnergy/Backend/API/TokenAPI.py`:

```python
JWT_SECRET_KEY = "super-secret-key"
```

Anyone with read access to the repository now knows the key used to sign every
JWT. The healer rewrites it to read from the environment:

```python
import os

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
```

The secret now lives only in the deployment environment, never in version
control.

## Components

The agent is made of two standard-library-only Python scripts plus this README.

### `scanner.py` — detect

Walks a directory, opens every `.py` file, and reports any line that assigns a
hardcoded secret matching this regular expression:

```python
r'(SECRET|KEY|PASSWORD)\s*=\s*["\'](super-secret-key|12345|password)["\']'
```

Run it against the repository root:

```bash
python scanner.py .
```

It prints each finding as `path:line: source` and exits with code `1` when
anything is found (useful for CI), or `0` when the codebase is clean.

### `healer.py` — repair

Reuses the scanner's detection rule, then rewrites each hardcoded literal to
`os.getenv("JWT_SECRET_KEY")`. It also guarantees the file imports `os`. Every
change is printed so you can review exactly what was modified:

```bash
python healer.py .
```

Example output:

```
[*] Healing hardcoded secrets under: /path/to/repo
[+] Fixed ./projects/PiNetAI/utils/constants.py:
  line 9: DB_PASSWORD = "password"  ->  DB_PASSWORD = os.getenv("JWT_SECRET_KEY")

[+] Done. Applied 1 fix(es).
[i] Remember to set the JWT_SECRET_KEY environment variable.
```

## Usage

From inside the `autonomous_healer_agent` folder:

```bash
# 1. Detect hardcoded secrets (read-only, safe to run anywhere).
python scanner.py /path/to/pi-nexus-autonomous-banking-network

# 2. Repair what was found.
python healer.py /path/to/pi-nexus-autonomous-banking-network
```

Both scripts default to scanning the current directory when no path is given.

## Design notes

The agent deliberately keeps things simple and predictable. It uses **only the
Python standard library** (`os`, `re`, `sys`), so it runs anywhere Python 3 is
available with no `pip install` step. It performs a **one-time scan** rather
than running continuously, which keeps it easy to reason about and safe to drop
into a pre-commit hook or a CI job. Directories such as `.git`, `__pycache__`,
and virtual environments are skipped automatically.

## After healing

Setting the environment variable is still required at deploy time:

```bash
export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

Rotate any secret that was previously committed — once a key has been pushed to
a remote, it should be considered compromised even after the code is fixed.

## Scope and limitations

This is a focused, pattern-based healer, not a full secret scanner. It only
catches the three placeholder values above; it does not detect high-entropy
keys, cloud provider credentials, or secrets stored in non-Python files. Treat
it as one layer in a defence-in-depth strategy alongside tools such as
`gitleaks`, `trufflehog`, and GitHub secret scanning.
