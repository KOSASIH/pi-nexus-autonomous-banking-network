# distributed_storage.py
import ipfshttpclient

class DistributedStorage:
    def __init__(self, ipfs_client):
        self.ipfs_client = ipfs_client

    def upload_file(self, file):
        # validate file
        self.ipfs_client.add(file)

    def download_file(self, file_hash):
        # validate file hash
        return self.ipfs_client.cat(file_hash)
