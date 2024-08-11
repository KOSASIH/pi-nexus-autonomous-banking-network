import Web3 from 'web3';

class BlockchainService {
  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
  }

  async getBlockNumber() {
    try {
      const blockNumber = await this.web3.eth.getBlockNumber();
      return blockNumber;
    } catch (error) {
      throw new Error(`Failed to get block number: ${error.message}`);
    }
  }

  async getTransactionCount(address) {
    try {
      const txCount = await this.web3.eth.getTransactionCount(address);
      return txCount;
    } catch (error) {
      throw new Error(`Failed to get transaction count: ${error.message}`);
    }
  }
}

export default BlockchainService;
