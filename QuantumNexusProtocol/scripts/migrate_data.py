import json
import requests

def migrate_data():
    # Load old data
    with open('old_data.json', 'r') as f:
        old_data = json.load(f)

    # Migrate data to new contract structure
    new_data = {}
    for key, value in old_data.items():
        new_data[key] = value  # Transform data as needed

    # Send new data to the blockchain
    response = requests.post("http://localhost:8545/migrate", json=new_data)
    if response.status_code == 200:
        print("Data migrated successfully.")
    else:
        print("Error migrating data:", response.status_code)

if __name__ == "__main__":
    migrate_data()
