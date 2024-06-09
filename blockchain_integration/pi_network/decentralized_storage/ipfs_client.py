import os
import json
from ipfshttpclient import connect

class IPFSClient:
    def __init__(self, ipfs_url: str = 'https://ipfs.io/ipfs/'):
        self.ipfs_url = ipfs_url
        self.client = connect(ipfs_url)

    def add_file(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            file_hash = self.client.add(file)
            return file_hash

    def get_file(self, file_hash: str) -> bytes:
        return self.client.cat(file_hash)

    def pin_file(self, file_hash: str) -> bool:
        return self.client.pin.add(file_hash)

    def unpin_file(self, file_hash: str) -> bool:
        return self.client.pin.rm(file_hash)

    def get_file_metadata(self, file_hash: str) -> dict:
        return self.client.object.get(file_hash)

    def search_files(self, query: str) -> list:
        return self.client.search(query)
