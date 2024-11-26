import subprocess

def update_contracts():
    print("Updating smart contracts...")
    try:
        subprocess.run(["truffle", "migrate", "--reset"], check=True)
        print("Smart contracts updated successfully.")
    except subprocess.CalledProcessError as e:
        print("Error updating contracts:", e)

if __name__ == "__main__":
    update_contracts()
