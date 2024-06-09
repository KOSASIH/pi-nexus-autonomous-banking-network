import os
import json
from swarm_python import Swarm

class SwarmClient:
    def __init__(self, swarm_url: str = 'https://swarm.pi.network/'):
        self.swarm_url = swarm_url
        self.swarm = Swarm(swarm_url)

    def upload_file(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            file_hash = self.swarm.upload(file)
            return file_hash

    def download_file(self, file_hash: str) -> bytes:
        return self.swarm.download(file_hash)

    def get_file_metadata(self, file_hash: str) -> dict:
        return self.swarm.get_file_metadata(file_hash)
