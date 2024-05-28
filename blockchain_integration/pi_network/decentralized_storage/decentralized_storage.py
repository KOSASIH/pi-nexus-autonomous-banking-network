# decentralized_storage.py

import os
import json
import hashlib
from typing import Dict, List

import ipfshttpclient

class DecentralizedStorage:
    def __init__(self, ipfs_api_url: str, ipfs_api_port: int):
        self.ipfs_client = ipfshttpclient.connect(f"/ip4/{ipfs_api_url}/tcp/{ipfs_api_port}/http")
        self.logger = logging.getLogger(__name__)

    def add_file(self, file_path: str) -> str:
        # Add a file to IPFS
        with open(file_path, "rb") as file:
            file_data = file.read()
            file_hash = hashlib.sha256(file_data).hexdigest()
            self.ipfs_client.add(file_data, wrap_with_directory=True)
            self.logger.info(f"Added file {file_path} to IPFS with hash {file_hash}")
            return file_hash

    def get_file(self, file_hash: str) -> bytes:
        # Retrieve a file from IPFS
        file_data = self.ipfs_client.cat(file_hash)
        self.logger.info(f"Retrieved file with hash {file_hash} from IPFS")
        return file_data

    def add_directory(self, directory_path: str) -> str:
        # Add a directory to IPFS
        directory_hash = self.ipfs_client.add(directory_path, recursive=True)
        self.logger.info(f"Added directory {directory_path} to IPFS with hash {directory_hash}")
        return directory_hash

    def get_directory(self, directory_hash: str) -> List[str]:
        # Retrieve a directory from IPFS
        directory_files = self.ipfs_client.ls(directory_hash)
        self.logger.info(f"Retrieved directory with hash {directory_hash} from IPFS")
        return [file["Hash"] for file in directory_files]

    def pin_file(self, file_hash: str) -> None:
        # Pin a file to IPFS to ensure it is stored permanently
        self.ipfs_client.pin.add(file_hash)
        self.logger.info(f"Pinned file with hash {file_hash} to IPFS")

    def unpin_file(self, file_hash: str) -> None:
        # Unpin a file from IPFS to allow it to be garbage collected
        self.ipfs_client.pin.rm(file_hash)
        self.logger.info(f"Unpinned file with hash {file_hash} from IPFS")

if __name__ == "__main__":
    config = Config()
    ipfs_api_url = config.get_ipfs_api_url()
    ipfs_api_port = config.get_ipfs_api_port()
    decentralized_storage = DecentralizedStorage(ipfs_api_url, ipfs_api_port)
    file_path = "example.txt"
    file_hash = decentralized_storage.add_file(file_path)
    file_data = decentralized_storage.get_file(file_hash)
    print(file_data.decode())
