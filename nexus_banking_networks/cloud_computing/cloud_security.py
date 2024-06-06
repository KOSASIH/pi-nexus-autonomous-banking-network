import boto3

iam = boto3.client('iam')
cloudhsm = boto3.client('cloudhsm')

def create_iam_role(role_name):
    # Create a new IAM role
    response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument='''{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "ec2.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }'''
    )
    return response['Role']['RoleId']

def create_cloudhsm_cluster(cluster_name):
    # Create a new CloudHSM cluster
    response = cloudhsm.create_cluster(
        ClusterName=cluster_name,
        HsmType='hsm1.medium'
    )
    return response['Cluster']['ClusterId']

def enable_encryption(key_id):
    # Enable encryption using AWS Key Management Service (KMS)
    kms = boto3.client('kms')
    response = kms.enable_key(
        KeyId=key_id
    )
    return response['KeyId']

if __name__ == '__main__':
    role_name = 'banking-role'
    cluster_name = 'banking-cluster'
    key_id = 'your-kms-key-id'

    role_id = create_iam_role(role_name)
    cluster_id = create_cloudhsm_cluster(cluster_name)
    enable_encryption(key_id)
    print(f"IAM role created with ID: {role_id}")
    print(f"CloudHSM cluster created with ID: {cluster_id}")
    print("Encryption enabled successfully!")
