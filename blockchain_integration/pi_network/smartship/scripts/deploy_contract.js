const { exec } = require('child_process');

const deployContract = () => {
  exec('truffle migrate --network development', (error, stdout, stderr) => {
    if (error) {
      console.error(`Error deploying contract: ${error}`);
      return;
    }
    console.log(`stdout: ${stdout}`);
    console.log(`stderr: ${stderr}`);
  });
};

deployContract();
