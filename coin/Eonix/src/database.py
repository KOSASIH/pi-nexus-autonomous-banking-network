# database.py
import ipfshttpclient

class EonixDatabase:
    def __init__(self):
        self.ipfs_client = ipfshttpclient.connect()

    def store_data(self, data):
        # Store data in IPFS
        return self.ipfs_client.add_json(data)

    def retrieve_data(self, cid):
        # Retrieve data from IPFS
        return self.ipfs_client.get_json(cid)
