// AdvancedOrder.js
import { BigNumber } from 'bignumber.js';
import { Web3Utils } from 'web3-utils';
import { abi, bytecode } from './contracts/AdvancedOrderContract';
import { getGasPrice, getBlockNumber } from './api/BlockchainAPI';

const AdvancedOrderContract = require('web3-eth-contract');

class AdvancedOrder {
  constructor(type, amount, price, stopLoss, takeProfit, leverage, expiration) {
    this.type = type; // 'limit', 'arket', 'top-loss', 'take-profit'
    this.amount = new BigNumber(amount);
    this.price = new BigNumber(price);
    this.stopLoss = new BigNumber(stopLoss);
    this.takeProfit = new BigNumber(takeProfit);
    this.leverage = leverage; // 1x, 2x, 5x, etc.
    this.expiration = expiration; // timestamp or block number
    this.contract = new AdvancedOrderContract(abi, bytecode);
  }

  async placeOrder() {
    const gasPrice = await getGasPrice();
    const blockNumber = await getBlockNumber();
    const txCount = await Web3Utils.getTransactionCount();

    const txData = this.contract.methods.placeOrder(
      this.type,
      this.amount.toFixed(),
      this.price.toFixed(),
      this.stopLoss.toFixed(),
      this.takeProfit.toFixed(),
      this.leverage,
      this.expiration
    ).encodeABI();

    const tx = {
      from: '0x...your_address...',
      to: this.contract.options.address,
      data: txData,
      gas: '20000',
      gasPrice: Web3Utils.toWei(gasPrice, 'gwei'),
      nonce: txCount,
    };

    try {
      const receipt = await Web3Utils.sendTransaction(tx);
      console.log(`Order placed successfully! Tx hash: ${receipt.transactionHash}`);
    } catch (error) {
      console.error(`Error placing order: ${error.message}`);
    }
  }

  async cancelOrder() {
    const gasPrice = await getGasPrice();
    const blockNumber = await getBlockNumber();
    const txCount = await Web3Utils.getTransactionCount();

    const txData = this.contract.methods.cancelOrder().encodeABI();

    const tx = {
      from: '0x...your_address...',
      to: this.contract.options.address,
      data: txData,
      gas: '20000',
      gasPrice: Web3Utils.toWei(gasPrice, 'gwei'),
      nonce: txCount,
    };

    try {
      const receipt = await Web3Utils.sendTransaction(tx);
      console.log(`Order cancelled successfully! Tx hash: ${receipt.transactionHash}`);
    } catch (error) {
      console.error(`Error cancelling order: ${error.message}`);
    }
  }

  async updateOrder(newPrice, newStopLoss, newTakeProfit) {
    const gasPrice = await getGasPrice();
    const blockNumber = await getBlockNumber();
    const txCount = await Web3Utils.getTransactionCount();

    const txData = this.contract.methods.updateOrder(
      newPrice.toFixed(),
      newStopLoss.toFixed(),
      newTakeProfit.toFixed()
    ).encodeABI();

    const tx = {
      from: '0x...your_address...',
      to: this.contract.options.address,
      data: txData,
      gas: '20000',
      gasPrice: Web3Utils.toWei(gasPrice, 'gwei'),
      nonce: txCount,
    };

    try {
      const receipt = await Web3Utils.sendTransaction(tx);
      console.log(`Order updated successfully! Tx hash: ${receipt.transactionHash}`);
    } catch (error) {
      console.error(`Error updating order: ${error.message}`);
    }
  }
}

export default AdvancedOrder;
