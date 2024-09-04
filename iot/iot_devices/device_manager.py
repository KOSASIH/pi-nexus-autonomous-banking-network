import json
import os
from typing import List, Dict

class DeviceManager:
    def __init__(self, device_db_file: str = "device_db.json"):
        self.device_db_file = device_db_file
        self.device_db: Dict[str, dict] = self.load_device_db()

    def load_device_db(self) -> Dict[str, dict]:
        if os.path.exists(self.device_db_file):
            with open(self.device_db_file, "r") as f:
                return json.load(f)
        else:
            return {}

    def save_device_db(self) -> None:
        with open(self.device_db_file, "w") as f:
            json.dump(self.device_db, f, indent=4)

    def add_device(self, device_id: str, device_info: dict) -> None:
        self.device_db[device_id] = device_info
        self.save_device_db()

    def get_device(self, device_id: str) -> dict:
        return self.device_db.get(device_id)

    def remove_device(self, device_id: str) -> None:
        if device_id in self.device_db:
            del self.device_db[device_id]
            self.save_device_db()

    def list_devices(self) -> List[str]:
        return list(self.device_db.keys())

# Example usage
manager = DeviceManager()
manager.add_device("device1", {"type": "temperature sensor", "location": "room 1"})
manager.add_device("device2", {"type": "humidity sensor", "location": "room 2"})
print(manager.list_devices())  # Output: ["device1", "device2"]
print(manager.get_device("device1"))  # Output: {"type": "temperature sensor", "location": "room 1"}
manager.remove_device("device1")
print(manager.list_devices())  # Output: ["device2"]
