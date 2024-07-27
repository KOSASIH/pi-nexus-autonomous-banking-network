from typing import Dict

class GreenBuildingContract:
    def __init__(self):
        self.building_owners: Dict[str, str] = {}
        self.green_buildings: Dict[str, Dict[str, str]] = {}

    def register_building(self, owner: str, building_id: str, building_type: str):
        # Register a new building
        self.building_owners[building_id] = owner
        self.green_buildings[building_id] = {
            "owner": owner,
            "building_type": building_type,
            "green_certifications": []
        }

    def add_green_certification(self, building_id: str, certification: str):
        # Add a green certification to a building
        if building_id not in self.green_buildings:
            raise ValueError("Building not found")
        self.green_buildings[building_id]["green_certifications"].append(certification)

    def remove_green_certification(self, building_id: str, certification: str):
        # Remove a green certification from a building
        if building_id not in self.green_buildings:
            raise ValueError("Building not found")
        self.green_buildings[building_id]["green_certifications"].remove(certification)

    def get_building_info(self, building_id: str):
        # Get the information of a building
        return self.green_buildings.get(building_id, {})

    def get_green_certifications(self, building_id: str):
        # Get the green certifications of a building
        return self.green_buildings.get(building_id, {}).get("green_certifications", [])

    def verify_green_building(self, building_id: str):
        # Verify that a building meets the green building standards
        building_info = self.get_building_info(building_id)
        if not building_info:
            return False
        green_certifications = self.get_green_certifications(building_id)
        if len(green_certifications) >= 3:
            return True
        return False
