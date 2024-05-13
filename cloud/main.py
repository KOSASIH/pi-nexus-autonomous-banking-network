import time
from cloud.ec2 import EC2
from cloud.s3 import S3

class Cloud:
    def __init__(self):
        self.ec2 = EC2("us-west-2")
        self.s3 = S3("us-west-2")

    def create_instance(self):
        image_id = "ami-0c94855ba95c574c8"
        instance_type = "t2.micro"
        key_name = "my-key-pair"
        instance = self.ec2.create_instance(image_id, instance_type, key_name)
        return instance

    def terminate_instance(self, instance):
        self.ec2.terminate_instance(instance)

    def create_bucket(self, bucket_name):
        self.s3.create_bucket(bucket_name)

    def upload_file(self, bucket_name, file_path, object_name):
        self.s3.upload_file(bucket_name, file_path, object_name)

    def download_file(self, bucket_name, object_name, file_path):
        self.s3.download_file(bucket_name, object_name, file_path)

if __name__ == "__main__":
    cloud = Cloud()

    # Menghidupkan instance EC2
    instance = cloud.create_instance()
    print(f"Instance ID: {instance.id}")

    # Menunggu beberapa saat sebelum mematikan instance
    time.sleep(60)

    # Mematikan instance EC2
    cloud.terminate_instance(instance)

    # Membuat bucket S3
    bucket_name = "my-banking-data"
    cloud.create_bucket(bucket_name)

    # Mengunggah file ke bucket S3
    file_path = "/path/to/my-file.txt"
    object_name = "my-file.txt"
    cloud.upload_file(bucket_name, file_path, object_name)

    # Mengunduh file dari bucket S3
    downloaded_file_path = "/path/to/downloaded-file.txt"
    cloud.download_file(bucket_name, object_name, downloaded_file_path)
