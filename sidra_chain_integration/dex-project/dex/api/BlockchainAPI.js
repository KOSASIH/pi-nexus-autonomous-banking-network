import Web3 from 'web3';
import AdvancedOrderContract from '../contracts/AdvancedOrderContract';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
const advancedOrderContract = new web3.eth.Contract(AdvancedOrderContract.abi, '0x...AdvancedOrderContractAddress...');

class BlockchainAPI {
  async placeOrder(amount, price, stopLoss, takeProfit, leverage, expiration) {
    const txCount = await web3.eth.getTransactionCount('0x...YourAccountAddress...');
    const txData = advancedOrderContract.methods.placeOrder(amount, price, stopLoss, takeProfit, leverage, expiration).encodeABI();
    const tx = {
      from: '0x...YourAccountAddress...',
      to: '0x...AdvancedOrderContractAddress...',
      data: txData,
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...YourAccountPrivateKey...');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    return receipt.transactionHash;
  }

  async cancelOrder(orderId) {
    const txCount = await web3.eth.getTransactionCount('0x...YourAccountAddress...');
    const txData = advancedOrderContract.methods.cancelOrder(orderId).encodeABI();
    const tx = {
      from: '0x...YourAccountAddress...',
      to: '0x...AdvancedOrderContractAddress...',
      data: txData,
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...YourAccountPrivateKey...');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    return receipt.transactionHash;
  }

  async updateOrder(orderId, newPrice, newStopLoss, newTakeProfit) {
    const txCount = await web3.eth.getTransactionCount('0x...YourAccountAddress...');
    const txData = advancedOrderContract.methods.updateOrder(orderId, newPrice, newStopLoss, newTakeProfit).encodeABI();
    const tx = {
      from: '0x...YourAccountAddress...',
      to: '0x...AdvancedOrderContractAddress...',
      data: txData,
      gas: '200000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...YourAccountPrivateKey...');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    return receipt.transactionHash;
  }

  async getOrders() {
    const orders = await advancedOrderContract.methods.getOrders().call();
    return orders;
  }

  async getOrder(orderId) {
    const order = await advancedOrderContract.methods.getOrder(orderId).call();
    return order;
  }
}

export default BlockchainAPI;
