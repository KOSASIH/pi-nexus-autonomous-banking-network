from typing import Dict

class ElectricVehicleChargingStationContract:
    def __init__(self):
        self.stations: Dict[str, Dict[str, str]] = {}
        self.owners: Dict[str, Dict[str, str]] = {}

    def register_station(self, station_id: str, owner_id: str, location: str, capacity: int):
        # Register a new electric vehicle charging station
        self.stations[station_id] = {
            "owner_id": owner_id,
            "location": location,
            "capacity": capacity,
            "available": capacity
        }
        self.owners[owner_id] = self.owners.get(owner_id, {})
        self.owners[owner_id][station_id] = station_id

    def update_station_availability(self, station_id: str, available: int):
        # Update the availability of a charging station
        if station_id not in self.stations:
            raise ValueError("Station not found")
        self.stations[station_id]["available"] = available

    def get_station_info(self, station_id: str):
        # Get the information of a charging station
        return self.stations.get(station_id, {})

    def get_owner_stations(self, owner_id: str):
        # Get the list of stations owned by an owner
        return list(self.owners.get(owner_id, {}).values())

    def charge_vehicle(self, station_id: str, vehicle_id: str, amount: int):
        # Charge a vehicle at a charging station
        if station_id not in self.stations:
            raise ValueError("Station not found")
        if self.stations[station_id]["available"] < amount:
            raise ValueError("Not enough capacity available")
        self.stations[station_id]["available"] -= amount
        # Update the vehicle's state (not implemented)

    def get_station_utilization(self, station_id: str):
        # Get the utilization of a charging station
        if station_id not in self.stations:
            raise ValueError("Station not found")
        return 1 - (self.stations[station_id]["available"] / self.stations[station_id]["capacity"])
