const { exec } = require('child_process');

const deploy = () => {
    exec('docker-compose up -d --build', (error, stdout, stderr) => {
        if (error) {
            console.error(`Error deploying: ${error.message}`);
            return;
        }
        if (stderr) {
            console.error(`Deployment stderr: ${stderr}`);
            return;
        }
        console.log(`Deployment stdout: ${stdout}`);
    });
};

deploy();
