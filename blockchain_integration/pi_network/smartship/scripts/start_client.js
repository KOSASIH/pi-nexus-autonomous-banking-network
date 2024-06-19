const { exec } = require('child_process');

const startClient = () => {
  exec('npm start', (error, stdout, stderr) => {
    if (error) {
      console.error(`Error starting client: ${error}`);
      return;
    }
    console.log(`stdout: ${stdout}`);
    console.log(`stderr: ${stderr}`);
  });
};

startClient();
