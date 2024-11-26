import subprocess

def run_security_audit():
    print("Running security audit...")
    try:
        subprocess.run(["safety", "check"], check=True)
        print("Security audit completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error during security audit:", e)

if __name__ == "__main__":
    run_security_audit()
