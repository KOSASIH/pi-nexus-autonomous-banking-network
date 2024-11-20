// setup.js
const fs = require('fs');
const path = require('path');
const inquirer = require('inquirer');

/**
 * Set up the project by creating necessary directories and files.
 */
async function setup() {
  try {
    // Create the necessary directories
    const dirsToCreate = ['src', 'public', 'deployments'];
    dirsToCreate.forEach((dir) => {
      fs.mkdirSync(path.join(__dirname, `../${dir}`), { recursive: true });
    });

    // Create the necessary files
    const filesToCreate = ['index.html', 'styles.css', 'script.js'];
    filesToCreate.forEach((file) => {
      fs.writeFileSync(path.join(__dirname, `../${file}`), '');
    });

    // Prompt the user for project information
    const answers = await inquirer.prompt([
      {
        type: 'input',
        name: 'projectName',
        message: 'What is the name of your project?',
      },
      {
        type: 'input',
        name: 'projectDescription',
        message: 'What is the description of your project?',
      },
    ]);

    // Create a package.json file with the project information
    const packageJson = {
      name: answers.projectName,
      description: answers.projectDescription,
      version: '1.0.0',
      scripts: {
        start: 'node script.js',
      },
    };
    fs.writeFileSync(path.join(__dirname, '../package.json'), JSON.stringify(packageJson, null, 2));

    console.log('Project setup complete!');
  } catch (error) {
    console.error('Error setting up project:', error);
  }
}

module.exports = setup;
