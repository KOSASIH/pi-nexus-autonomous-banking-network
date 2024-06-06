import boto3

s3 = boto3.client('s3')
glacier = boto3.client('glacier')

def store_data_in_s3(bucket_name, data):
    # Store data in S3 bucket
    s3.put_object(Body=data, Bucket=bucket_name, Key='data.txt')

def archive_data_in_glacier(vault_name, data):
    # Archive data in Glacier vault
    archive_id = glacier.upload_archive(
        vaultName=vault_name,
        archiveDescription='Banking data archive',
        body=data
    )['archiveId']
    return archive_id

if __name__ == '__main__':
    bucket_name = 'your-s3-bucket-name'
    vault_name = 'your-glacier-vault-name'
    data = b'Your banking data here...'

    store_data_in_s3(bucket_name, data)
    archive_id = archive_data_in_glacier(vault_name, data)
    print(f"Data archived in Glacier with ID: {archive_id}")
