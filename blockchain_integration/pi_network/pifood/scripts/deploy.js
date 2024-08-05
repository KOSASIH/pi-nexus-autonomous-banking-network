// deploy.js
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { argv } from 'yargs';
import { getEnv } from '../config/env';
import { getDeployConfig } from '../config/deploy';

const { env, deploy } = getEnv();
const { bucket, region, accessKeyId, secretAccessKey } = getDeployConfig();

const deployToS3 = () => {
  console.log('Deploying to S3...');
  const s3Cmd = `aws s3 cp ${path.resolve(__dirname, '../dist')} s3://${bucket} --recursive --region ${region} --access-key-id ${accessKeyId} --secret-access-key ${secretAccessKey}`;
  exec(s3Cmd, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error deploying to S3: ${error}`);
      return;
    }
    console.log(`Deployed to S3 successfully!`);
  });
};

const deployToCloudFront = () => {
  console.log('Deploying to CloudFront...');
  const cloudFrontCmd = `aws cloudfront create-invalidation --distribution-id ${deploy.cloudFrontDistributionId} --paths "/*"`;
  exec(cloudFrontCmd, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error deploying to CloudFront: ${error}`);
      return;
    }
    console.log(`Deployed to CloudFront successfully!`);
  });
};

const deployToLambda = () => {
  console.log('Deploying to Lambda...');
  const lambdaCmd = `aws lambda update-function-code --function-name ${deploy.lambdaFunctionName} --zip-file fileb://${path.resolve(__dirname, '../lambda.zip')}`;
  exec(lambdaCmd, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error deploying to Lambda: ${error}`);
      return;
    }
    console.log(`Deployed to Lambda successfully!`);
  });
};

const deployAll = () => {
  deployToS3();
  deployToCloudFront();
  deployToLambda();
};

if (argv.env === 'prod') {
  deployAll();
} else {
  console.log('Not deploying to production environment');
}

export default deployAll;
