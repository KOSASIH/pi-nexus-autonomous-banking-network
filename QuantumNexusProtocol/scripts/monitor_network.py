import time
import requests

def monitor_network():
    while True:
        try:
            response = requests.get("http://localhost:8545/status")
            if response.status_code == 200:
                print("Network is up and running.")
            else:
                print("Network is down! Status code:", response.status_code)
        except requests.exceptions.RequestException as e:
            print("Error connecting to the network:", e)

        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    monitor_network()
