const { ethers } = require("ethers");

class EthClient {
  constructor() {
    this.provider = new ethers.providers.JsonRpcProvider("http://localhost:8545");
    this.wallet = new ethers.Wallet("myPrivateKey", this.provider);
    this.signer = this.wallet.connect(this.provider);
  }

  async getBalance(address) {
    const balance = await this.provider.getBalance(address);
    return ethers.utils.formatEther(balance);
  }

  async sendTransaction(to, value) {
    const tx = await this.signer.sendTransaction({
      to: to,
      value: ethers.utils.parseEther(value),
    });

    console.log(`Transaction sent: ${tx.hash}`);

    return tx;
  }

  async getTransactionReceipt(txHash) {
    const receipt = await this.provider.getTransactionReceipt(txHash);
    return receipt;
  }

  async getPastEvents(contract, eventName, fromBlock, toBlock) {
    const eventFilter = contract.filters[eventName]();
    const events = await contract.queryFilter(eventFilter, fromBlock, toBlock);
    return events;
  }
}

module.exports = EthClient;
