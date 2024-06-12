import ipfshttpclient

class DecentralizedDataStorage:
    def __init__(self, ipfs_client):
        self.ipfs_client = ipfs_client

    def store_data(self, data):
        cid = self.ipfs_client.add(data)
        return cid

    def retrieve_data(self, cid):
        data = self.ipfs_client.cat(cid)
        return data

# Example usage:
ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
data_storage = DecentralizedDataStorage(ipfs_client)
data = b'Hello, Decentralized World!'
cid = data_storage.store_data(data)
print(cid)
retrieved_data = data_storage.retrieve_data(cid)
print(retrieved_data)
