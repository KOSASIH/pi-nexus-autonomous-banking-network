import time
import requests
from typing import Dict, Any

class Forks:
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize the Forks class with a dictionary of API keys.
        """
        self.api_keys = api_keys

    def get_forks(self) -> Dict[str, Any]:
"""
        Get a list of forks for the Pi-Nexus Autonomous Banking Network project.
        """
        url = "https://api.github.com/repos/KOSASIH/pi-nexus-autonomous-banking-network/forks"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['github']}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def create_fork(self) -> Dict[str, Any]:
        """
        Create a new fork of the Pi-Nexus Autonomous Banking Network project.
        """
        url = "https://api.github.com/repos/KOSASIH/pi-nexus-autonomous-banking-network/forks"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['github']}"
        }
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        return response.json()

# Example usage
api_keys = {
    "github": "your_github_api_key"
}
forks = Forks(api_keys)

# Get a list of forks for the Pi-Nexus Autonomous Banking Network project
forks_list = forks.get_forks()
print(forks_list)

# Create a new fork of the Pi-Nexus Autonomous Banking Network project
new_fork = forks.create_fork()
print(new_fork)
