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

export async function delete
