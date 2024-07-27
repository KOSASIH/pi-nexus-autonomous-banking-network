from typing import Dict

class ElectricVehicleIncentiveContract:
    def __init__(self):
        self.vehicle_owners: Dict[str, Dict[str, str]] = {}
        self.incentives: Dict[str, Dict[str, str]] = {}

    def register_vehicle(self, owner_id: str, vehicle_id: str, vehicle_type: str):
        # Register a new electric vehicle
        self.vehicle_owners[owner_id] = {
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "incentives": []
        }

    def add_incentive(self, owner_id: str, incentive_id: str, incentive_type: str, amount: float):
        # Add an incentive to a vehicle owner
        if owner_id not in self.vehicle_owners:
            raise ValueError("Owner not found")
        self.vehicle_owners[owner_id]["incentives"].append(incentive_id)
        self.incentives[incentive_id] = {
            "owner_id": owner_id,
            "incentive_type": incentive_type,
            "amount": amount
        }

    def remove_incentive(self, owner_id: str, incentive_id: str):
        # Remove an incentive from a vehicle owner
        if owner_id not in self.vehicle_owners:
            raise ValueError("Owner not found")
        if incentive_id not in self.vehicle_owners[owner_id]["incentives"]:
            raise ValueError("Incentive not found")
        self.vehicle_owners[owner_id]["incentives"].remove(incentive_id)
        del self.incentives[incentive_id]

    def get_vehicle_info(self, owner_id: str):
        # Get the information of a vehicle
        return self.vehicle_owners.get(owner_id, {})

    def get_incentive_info(self, incentive_id: str):
        # Get the information of an incentive
        return self.incentives.get(incentive_id, {})

    def get_total_incentives(self, owner_id: str):
        # Get the total incentives for a vehicle owner
        total_incentives = 0
        for incentive_id in self.vehicle_owners[owner_id]["incentives"]:
            total_incentives += self.incentives[incentive_id]["amount"]
        return total_incentives
