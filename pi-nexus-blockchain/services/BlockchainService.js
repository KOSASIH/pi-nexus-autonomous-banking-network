import Web3 from 'web3';
import { NETWORK_ID, GAS_PRICE, GAS_LIMIT } from '../utils/constants';
import { BlockchainError } from '../utils/errors';

class BlockchainService {
  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider(`https://mainnet.infura.io/v3/YOUR_PROJECT_ID`));
  }

  /**
   * Get the current block number
   * @returns {Promise<number>} The current block number
   */
  async getBlockNumber() {
    try {
      const blockNumber = await this.web3.eth.getBlockNumber();
      return blockNumber;
    } catch (error) {
      throw new BlockchainError(`Failed to get block number: ${error.message}`);
    }
  }

  /**
   * Get the transaction count for a given address
   * @param {string} address The address to get the transaction count for
   * @returns {Promise<number>} The transaction count for the given address
   */
  async getTransactionCount(address) {
    try {
      const txCount = await this.web3.eth.getTransactionCount(address);
      return txCount;
    } catch (error) {
      throw new BlockchainError(`Failed to get transaction count: ${error.message}`);
    }
  }

  /**
   * Get the balance of a given address
   * @param {string} address The address to get the balance for
   * @returns {Promise<string>} The balance of the given address in wei
   */
  async getBalance(address) {
    try {
      const balance = await this.web3.eth.getBalance(address);
      return balance;
    } catch (error) {
      throw new BlockchainError(`Failed to get balance: ${error.message}`);
    }
  }

  /**
   * Send a transaction to the blockchain
   * @param {string} from The address to send the transaction from
   * @param {string} to The address to send the transaction to
   * @param {string} value The value to send in wei
   * @param {string} gasPrice The gas price to use in wei
   * @param {string} gasLimit The gas limit to use
   * @returns {Promise<string>} The transaction hash
   */
  async sendTransaction(from, to, value, gasPrice = GAS_PRICE, gasLimit = GAS_LIMIT) {
    try {
      const txCount = await this.getTransactionCount(from);
      const tx = {
        from,
        to,
        value,
        gas: gasLimit,
        gasPrice,
        nonce: txCount,
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...privateKey...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt.transactionHash;
    } catch (error) {
      throw new BlockchainError(`Failed to send transaction: ${error.message}`);
    }
  }

  /**
   * Get the transaction receipt for a given transaction hash
   * @param {string} transactionHash The transaction hash to get the receipt for
   * @returns {Promise<object>} The transaction receipt
   */
  async getTransactionReceipt(transactionHash) {
    try {
      const receipt = await this.web3.eth.getTransactionReceipt(transactionHash);
      return receipt;
    } catch (error) {
      throw new BlockchainError(`Failed to get transaction receipt: ${error.message}`);
    }
  }
}

export default BlockchainService;
