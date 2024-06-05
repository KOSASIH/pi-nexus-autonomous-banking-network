import boto3

class AWSLambda:
    def __init__(self, function_name):
        self.function_name = function_name
        self.lambda_client = boto3.client('lambda')

    def invoke_function(self, event):
        # Invoke AWS Lambda function
        response = self.lambda_client.invoke(
            FunctionName=self.function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(event)
        )
        return response
