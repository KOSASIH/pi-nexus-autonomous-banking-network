exports.handler = async (event) => {
  const { AWS } = require('aws-sdk');
  const lambda = new AWS.Lambda({ region: 'us-west-2' });
  const apiGateway = new AWS.APIGateway({ region: 'us-west-2' });

  // Create a new Lambda function
  const func = await lambda.createFunction({
    FunctionName: 'banking-function',
    Runtime: 'nodejs14.x',
    Handler: 'index.handler',
    Role: 'arn:aws:iam::123456789012:role/lambda-execution-role',
    Code: {
      ZipFile: fs.readFileSync('lambda_function.zip')
    }
  }).promise();

  // Create a new API Gateway REST API
  const restApi = await apiGateway.createRestApi({
    name: 'Banking API',
    description: 'API for banking operations'
  }).promise();

  // Create a new API Gateway resource and method
  const resource = await apiGateway.createResource({
    restApiId: restApi.id,
    parentId: restApi.rootResourceId,
    pathPart: 'accounts'
  }).promise();

  const method = await apiGateway.putMethod({
    restApiId: restApi.id,
    resourceId: resource.id,
    httpMethod: 'GET',
    authorization: 'NONE'
  }).promise();

  // Integrate Lambda function with API Gateway
  await apiGateway.putIntegration({
    restApiId: restApi.id,
    resourceId: resource.id,
    httpMethod: 'GET',
    integrationHttpMethod: 'POST',
    type: 'LAMBDA',
    uri: `arn:aws:apigateway:us-west-2:lambda:path/2015-03-31/functions/${func.FunctionArn}/invocations`
  }).promise();

  return { statusCode: 201, body:'Cloud function created successfully!' };
};
