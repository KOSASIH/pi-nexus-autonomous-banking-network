import os
import json
import boto3
from typing import Dict, List
from google.cloud import storage
from azure.storage.blob import BlobServiceClient

class CloudFunctions:
    def __init__(self, cloud_provider: str):
        self.cloud_provider = cloud_provider
        if cloud_provider == "aws":
            self.s3 = boto3.client("s3")
        elif cloud_provider == "gcp":
            self.storage_client = storage.Client()
        elif cloud_provider == "azure":
            self.blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=<account_name>;AccountKey=<account_key>;BlobEndpoint=<blob_endpoint>")

    def upload_file(self, file_path: str, bucket_name: str, object_name: str) -> None:
        if self.cloud_provider == "aws":
            self.s3.upload_file(file_path, bucket_name, object_name)
        elif self.cloud_provider == "gcp":
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            blob.upload_from_filename(file_path)
        elif self.cloud_provider == "azure":
            blob_client = self.blob_service_client.get_blob_client(bucket_name, object_name)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        if self.cloud_provider == "aws":
            self.s3.download_file(bucket_name, object_name, file_path)
        elif self.cloud_provider == "gcp":
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            blob.download_to_filename(file_path)
        elif self.cloud_provider == "azure":
            blob_client = self.blob_service_client.get_blob_client(bucket_name, object_name)
            with open(file_path, "wb") as data:
                blob_client.download_blob().readinto(data)

    def list_buckets(self) -> List[str]:
        if self.cloud_provider == "aws":
            response = self.s3.list_buckets()
            return [bucket["Name"] for bucket in response["Buckets"]]
        elif self.cloud_provider == "gcp":
            buckets = self.storage_client.list_buckets()
            return [bucket.name for bucket in buckets]
        elif self.cloud_provider == "azure":
            return [blob.name for blob in self.blob_service_client.list_containers()]

    def create_bucket(self, bucket_name: str) -> None:
        if self.cloud_provider == "aws":
            self.s3.create_bucket(Bucket=bucket_name)
        elif self.cloud_provider == "gcp":
            bucket = self.storage_client.create_bucket(bucket_name)
        elif self.cloud_provider == "azure":
            self.blob_service_client.create_container(bucket_name)

# Example usage:
cloud_functions = CloudFunctions("aws")
cloud_functions.upload_file("path/to/file.txt", "my-bucket", "file.txt")
cloud_functions.download_file("my-bucket", "file.txt", "path/to/downloaded_file.txt")
print(cloud_functions.list_buckets())
cloud_functions.create_bucket("new-bucket")
