import subprocess

def optimize_performance():
    print("Optimizing network performance...")
    try:
        subprocess.run(["npm", "run", "optimize"], check=True)
        print("Performance optimization completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error during performance optimization:", e)

if __name__ == "__main__":
    optimize_performance()
