// deploy.js
const fs = require('fs');
const path = require('path');
const childProcess = require('child_process');

/**
 * Deploy the application to a production environment.
 * @param {string} env - The environment to deploy to (e.g., prod, staging).
 */
async function deploy(env) {
  try {
    // Create a new directory for the deployment
    const deployDir = path.join(__dirname, `../deployments/${env}`);
    fs.mkdirSync(deployDir, { recursive: true });

    // Copy the necessary files to the deployment directory
    const filesToCopy = ['index.html', 'styles.css', 'script.js'];
    filesToCopy.forEach((file) => {
      fs.copyFileSync(path.join(__dirname, `../${file}`), path.join(deployDir, file));
    });

    // Run the deployment script
    childProcess.execSync(`npm run deploy:${env}`, { cwd: deployDir });

    console.log(`Deployment to ${env} environment successful!`);
  } catch (error) {
    console.error(`Error deploying to ${env} environment:`, error);
  }
}

module.exports = deploy;
