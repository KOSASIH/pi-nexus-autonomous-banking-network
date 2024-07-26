# dex_project_cloud_deployer.py
import os
from google.cloud import storage

class DexProjectCloudDeployer:
    def __init__(self, project_id, bucket_name):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client(project=self.project_id)

    def deploy_to_cloud(self, file_path):
        # Deploy file to cloud storage
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        return blob.public_url
