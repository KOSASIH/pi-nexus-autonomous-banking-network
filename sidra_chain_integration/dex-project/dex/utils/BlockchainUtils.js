import Web3 from 'web3';

class BlockchainUtils {
  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
  }

  async getBlockNumber() {
    // Implement advanced block number retrieval logic here
    const blockNumber = await this.web3.eth.getBlockNumber();
    return blockNumber;
  }

  async getTransactionCount(address) {
    // Implement advanced transaction count retrieval logic here
    const txCount = await this.web3.eth.getTransactionCount(address);
    return txCount;
  }
}

export default BlockchainUtils;
