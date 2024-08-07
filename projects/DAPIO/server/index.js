const express = require('express');
const app = express();
const Web3 = require('web3');
const { Dapio } = require('./Dapio');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const dapioContract = new web3.eth.Contract(Dapio.abi, '0x...DapioContractAddress...');

app.use(express.json());

app.post('/data-feeds', async (req, res) => {
  const { dataFeedName, dataFeedDescription } = req.body;
  const tx = await dapioContract.methods.createDataFeed(dataFeedName, dataFeedDescription).send({ from: '0x...YourEthereumAddress...' });
  res.json({ txHash: tx.transactionHash });
});

app.post('/ai-models', async (req, res) => {
  const { aiModelType, dataFeedAddress } = req.body;
  const tx = await dapioContract.methods.trainAiModel(aiModelType, dataFeedAddress).send({ from: '0x...YourEthereumAddress...' });
  res.json({ txHash: tx.transactionHash });
});

app.put('/data-feeds/:id', async (req, res) => {
  const { id } = req.params;
  const { dataFeedName, dataFeedDescription } = req.body;
  const tx = await dapioContract.methods.updateDataFeed(id, dataFeedName, dataFeedDescription).send({ from: '0x...YourEthereumAddress...' });
  res.json({ txHash: tx.transactionHash });
});

app.delete('/data-feeds/:id', async (req, res) => {
  const { id } = req.params;
  const tx = await dapioContract.methods.deleteDataFeed(id).send({ from: '0x...YourEthereumAddress...' });
  res.json({ txHash: tx.transactionHash });
});

app.get('/data-feeds', async (req, res) => {
  const dataFeeds = await dapioContract.methods.getDataFeeds().call();
  res.json(dataFeeds);
});

app.get('/ai-models', async (req, res) => {
  const aiModels = await dapioContract.methods.getAiModels().call();
  res.json(aiModels);
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
