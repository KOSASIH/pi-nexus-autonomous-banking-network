// app.js
const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const Web3 = require('web3');
const contract = require('../contracts/identity-verification-contract.json');
const machineLearningVerification = require('../machine-learning-identity-verification');

app.use(bodyParser.json());

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const contractInstance = new web3.eth.Contract(contract.abi, contract.networks['5777'].address);

app.post('/api/verify', async (req, res) => {
  try {
    const { userAddress, userData } = req.body;
    const isVerifiedML = await machineLearningVerification.verifyUserML(userData);

    if (isVerifiedML) {
      await contractInstance.methods.verifyUser(userAddress).send({ from: '0xYOUR_OWNER_ADDRESS' });
      res.json({ verified: true });
    } else {
      res.json({ verified: false });
    }
  } catch (error) {
    console.error(`Error verifying user: ${error}`);
    res.status(500).json({ error: 'Error verifying user' });
  }
});

app.listen(3001, () => {
  console.log('Backend API listening on port 3001');
});
