import boto3

apigateway_client = boto3.client('apigateway')

def create_api():
    # Create API Gateway REST API
    response = apigateway_client.create_rest_api(
        name='Pi Nexus API',
        description='Pi Nexus API Gateway'
    )
    api_id = response['id']

    # Create API Gateway resource and method
    response = apigateway_client.create_resource(
        restApiId=api_id,
        parentId='root',
        pathPart='pi-nexus'
    )
    resource_id = response['id']

    response = apigateway_client.put_method(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='GET',
        authorization='NONE'
    )

    # Create API Gateway integration with Lambda function
    response = apigateway_client.put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='GET',
        integrationHttpMethod='POST',
        type='LAMBDA',
        uri='arn:aws:lambda:REGION:ACCOUNT_ID:function:pi-nexus-lambda'
    )

    return api_id

def deploy_api(api_id):
    # Deploy API Gateway API
    response = apigateway_client.create_deployment(
        restApiId=api_id,
        stageName='prod'
    )
    deployment_id = response['id']

    return deployment_id
