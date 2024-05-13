import boto3

class EC2:
    def __init__(self, region):
        self.ec2 = boto3.resource('ec2', region_name=region)

    def create_instance(self, image_id, instance_type, key_name):
        instance = self.ec2.create_instances(
            ImageId=image_id,
            MinCount=1,
            MaxCount=1,
            InstanceType=instance_type,
            KeyName=key_name
        )
        instance.wait_until_running()
        instance.reload()
        return instance

    def terminate_instance(self, instance):
        instance.terminate()
        instance.wait_until_terminated()
