import boto3


class S3:
    def __init__(self, region):
        self.s3 = boto3.resource("s3", region_name=region)

    def create_bucket(self, bucket_name):
        self.s3.create_bucket(Bucket=bucket_name)

    def upload_file(self, bucket_name, file_path, object_name):
        self.s3.Bucket(bucket_name).upload_file(file_path, object_name)

    def download_file(self, bucket_name, object_name, file_path):
        self.s3.Bucket(bucket_name).download_file(object_name, file_path)
