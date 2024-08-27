import boto3

lambda_client = boto3.client('lambda')

def handler(event, context):
    # Process event and return response
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Hello from Lambda!'})
    }
