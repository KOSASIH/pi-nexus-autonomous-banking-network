import chalk from "chalk";
import fs from "fs";
import path from "path";
import { exec } from "child_process";
import { prompt } from "inquirer";
import { getEnvConfig } from "../config/env";
import { getDeployConfig } from "../config/deploy";
import { getGitBranch } from "../utils/git";

const deploy = async () => {
  const envConfig = getEnvConfig();
  const deployConfig = getDeployConfig();
  const gitBranch = await getGitBranch();

  console.log(chalk.cyan(`Deploying to ${deployConfig.environment} environment...`));

  // Check if branch is allowed to deploy
  if (!deployConfig.allowedBranches.includes(gitBranch)) {
    console.error(chalk.red(`Error: Branch ${gitBranch} is not allowed to deploy`));
    process.exit(1);
  }

  // Prompt for confirmation
  const response = await prompt([
    {
      type: "confirm",
      name: "confirm",
      message: `Are you sure you want to deploy to ${deployConfig.environment} environment?`,
    },
  ]);

  if (!response.confirm) {
    console.log(chalk.yellow("Deployment cancelled"));
    process.exit(0);
  }

  // Build and package application
  console.log(chalk.cyan("Building and packaging application..."));
  exec("npm run build", (error, stdout, stderr) => {
    if (error) {
      console.error(chalk.red(`Error: ${error.message}`));
      process.exit(1);
    }
    console.log(stdout);
  });

  // Upload to cloud storage
  console.log(chalk.cyan("Uploading to cloud storage..."));
  const storageBucket = envConfig.storageBucket;
  const filePath = path.join(__dirname, "../dist", "app.zip");
  const fileBuffer = fs.readFileSync(filePath);
  const uploadOptions = {
    Bucket: storageBucket,
    Key: "app.zip",
    Body: fileBuffer,
  };
  const s3 = new AWS.S3({ region: envConfig.region });
  s3.upload(uploadOptions, (error, data) => {
    if (error) {
      console.error(chalk.red(`Error: ${error.message}`));
      process.exit(1);
    }
    console.log(chalk.green(`Uploaded to ${storageBucket}`));
  });

  // Deploy to cloud platform
  console.log(chalk.cyan("Deploying to cloud platform..."));
  const cloudPlatform = envConfig.cloudPlatform;
  const deploymentOptions = {
    platform: cloudPlatform,
    environment: deployConfig.environment,
    storageBucket,
  };
  const deployment = new Deployment(deploymentOptions);
  deployment.deploy((error, data) => {
    if (error) {
      console.error(chalk.red(`Error: ${error.message}`));
      process.exit(1);
    }
    console.log(chalk.green(`Deployed to ${cloudPlatform}`));
  });
};

deploy();
