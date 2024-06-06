import boto3

lambda_client = boto3.client('lambda')
apigateway = boto3.client('apigateway')

def create_lambda_function(function_name):
    # Create a new Lambda function
    response = lambda_client.create_function(
        FunctionName=function_name,
        Runtime='python3.8',
        Role='arn:aws:iam::123456789012:role/lambda-execution-role',
        Handler='index.handler'
    )
    return response['FunctionArn']

def create_api_gateway_rest_api(api_name):
    # Create a new API Gateway REST API
    response = apigateway.create_rest_api(
        name=api_name,
        description='Banking API'
    )
    return response['id']

def create_api_gateway_resource(api_id, resource_name):
    # Create a new API Gateway resource
    response = apigateway.create_resource(
        restApiId=api_id,
        parentId='root',
        pathPart=resource_name
    )
    return response['id']

if __name__ == '__main__':
    function_name = 'banking-lambda-function'
    api_name = 'banking-api'
    resource_name = 'accounts'

    function_arn = create_lambda_function(function_name)
    api_id = create_api_gateway_rest_api(api_name)
    resource_id = create_api_gateway_resource(api_id, resource_name)
    print(f"Lambda function created with ARN: {function_arn}")
    print(f"API Gateway REST API created with ID: {api_id}")
    print(f"API Gateway resource created with ID: {resource_id}")
