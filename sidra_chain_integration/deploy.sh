#!/bin/bash

# Create an S3 bucket
aws s3api create-bucket --bucket sidra-lending-platform-frontend --region us-east-1

# Create an API Gateway
aws apigateway create-rest-api --name sidra-lending-platform-api --description "Sidra Lending Platform API"

# Create a Lambda function
aws lambda create-function --function-name sidra-lending-platform-lambda --runtime nodejs14.x --handler index.handler --role lambda-execution-role --zip-file fileb://lambda-function.zip

# Create a DynamoDB table
aws dynamodb create-table --table-name sidra-lending-platform-data --attribute-definitions AttributeName=creditScore,AttributeType=N AttributeName=income,AttributeType=N AttributeName=employmentHistory,AttributeType=S --key-schema AttributeName=creditScore,KeyType=HASH --table-status ACTIVE --region us-east-1

# Deploy frontend code to S3
aws s3 cp frontend/index.html s3://sidra-lending-platform-frontend/index.html
aws s3 cp frontend/script.js s3://sidra-lending-platform-frontend/script.js
aws s3 cp frontend/styles.css s3://sidra-lending-platform-frontend/styles.css

# Configure API Gateway
aws apigateway put-integration --rest-api-id <API_ID> --resource-id <RESOURCE_ID> --http-method POST --integration-http-method POST --type LAMBDA --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:<ACCOUNT_ID>:function:sidra-lending-platform-lambda/invocations

# Test the platform
curl https://<API_ID>.execute-api.us-east-1.amazonaws.com/loan-processing
