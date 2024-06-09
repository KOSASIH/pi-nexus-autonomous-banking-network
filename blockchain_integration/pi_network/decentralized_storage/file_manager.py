from decentralized_storage import DecentralizedStorage

class FileManager:
    def __init__(self, decentralized_storage: DecentralizedStorage):
        self.decentralized_storage = decentralized_storage

    def upload_file(self, file_path: str) -> str:
        return self.decentralized_storage.store_file(file_path)

    def download_file(self, file_hash: str) -> bytes:
        return self.decentralized_storage.retrieve_file(file_hash)

    def get_file_metadata(self, file_hash: str) -> dict:
        return self.decentralized_storage.get_file_metadata(file_hash)
