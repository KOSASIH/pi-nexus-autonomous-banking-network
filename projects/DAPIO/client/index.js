import Web3 from 'web3';
import { Dapio } from './Dapio';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const dapioContract = new web3.eth.Contract(Dapio.abi, '0x...DapioContractAddress...');

export async function createDataFeed(dataFeedName, dataFeedDescription) {
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: '0x...YourEthereumAddress...',
    to: dapioContract.address,
    value: '0',
    gas: '200000',
    gasPrice: '20',
    nonce: txCount,
    data: dapioContract.methods.createDataFeed(dataFeedName, dataFeedDescription).encodeABI(),
  };

  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...YourEthereumPrivateKey...');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

  return receipt;
}

export async function trainAiModel(aiModelType, dataFeedAddress) {
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: '0x...YourEthereumAddress...',
    to: dapioContract.address,
    value: '0',
    gas: '200000',
    gasPrice: '20',
    nonce: txCount,
    data: dapioContract.methods.trainAiModel(aiModelType, dataFeedAddress).encodeABI(),
  };

  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...YourEthereumPrivateKey...');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

  return receipt;
}

export async function updateDataFeed(dataFeedAddress, dataFeedName, dataFeedDescription) {
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: '0x...YourEthereumAddress...',
    to: dapioContract.address,
    value: '0',
    gas: '200000',
    gasPrice: '20',
    nonce: txCount,
    data: dapioContract.methods.updateDataFeed(dataFeedAddress, dataFeedName, dataFeedDescription).encodeABI(),
  };

  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...YourEthereumPrivateKey...');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

  return receipt;
}

export async function deleteDataFeed(dataFeedAddress) {
  const txCount = await web3.eth.getTransactionCount();
  const tx = {
    from: '0x...YourEthereumAddress...',
    to: dapioContract.address,
    value: '0',
    gas: '200000',
    gasPrice: '20',
    nonce: txCount,
    data: dapioContract.methods.deleteDataFeed(dataFeedAddress).encodeABI(),
  };

  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...YourEthereumPrivateKey...');
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

  return receipt;
}

export async function getDataFeed(dataFeedAddress) {
  const dataFeed = await dapioContract.methods.getDataFeed(dataFeedAddress).call();
  return dataFeed;
}

export async function getAiModel(aiModelAddress) {
  const aiModel = await dapioContract.methods.getAiModel(aiModelAddress).call();
  return aiModel;
}

export async function getAiModelForDataFeed(dataFeedAddress) {
  const aiModelAddress = await dapioContract.methods.getAiModelForDataFeed(dataFeedAddress).call();
  return aiModelAddress;
}

export async function getDataFeeds() {
  const dataFeeds = await dapioContract.methods.getDataFeeds().call();
  return dataFeeds;
}

export async function getAiModels() {
  const aiModels = await dapioContract.methods.getAiModels().call();
  return aiModels;
}
