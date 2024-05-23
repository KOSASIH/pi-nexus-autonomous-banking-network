import time

import boto3

# Set AWS credentials
aws_access_key_id = "your_aws_access_key_id"
aws_secret_access_key = "your_aws_secret_access_key"

# Initialize AWS DRaaS client
dr_client = boto3.client(
    "drs",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name="us-west-2",
)

# Define function to create a DRaaS replication job


def create_replication_job(source_server_id, destination_server_id):
    # Set DRaaS replication job parameters
    job_params = {
        "SourceServerID": source_server_id,
        "DestinationServerID": destination_server_id,
        "ReplicationSettings": {
            "ReplicationServerID": "your_replication_server_id",
            "RecoveryPointTags": [
                {"Key": "Name", "Value": "Pi-Nexus-DRaaS-Replication"}
            ],
        },
    }

    # Create DRaaS replication job
    dr_client.start_replication_run(**job_params)

    # Wait for replication job to complete
    while True:
        job_status = dr_client.describe_replication_runs(
            ReplicationRunIDs=[job_params["ReplicationRunID"]]
        )["ReplicationRuns"][0]["Status"]

        if job_status == "COMPLETED":
            print("DRaaS replication job completed successfully!")
            break
        elif job_status == "FAILED":
            print("DRaaS replication job failed!")
            break

        time.sleep(60)


# Example usage
source_server_id = "your_source_server_id"
destination_server_id = "your_destination_server_id"

create_replication_job(source_server_id, destination_server_id)
