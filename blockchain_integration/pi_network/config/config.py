# config.py
coins_file = "coins.json"
stable_value = 10.0
exchanges = ["localhost:8080"]

# Load the configuration from a file
with open("config.json", "r") as file:
    config = json.load(file)

# Set the configuration variables
coins_file = config["coins_file"]
stable_value = config["stable_value"]
exchanges = config["exchanges"]
