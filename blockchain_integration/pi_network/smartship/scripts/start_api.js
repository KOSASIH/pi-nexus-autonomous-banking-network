const { exec } = require('child_process');

const startApi = () => {
  exec('node api.js', (error, stdout, stderr) => {
    if (error) {
      console.error(`Error starting API: ${error}`);
      return;
    }
    console.log(`stdout: ${stdout}`);
    console.log(`stderr: ${stderr}`);
  });
};

startApi();
