const Web3 = require('web3');
const Ethers = require('ethers');

class NexusGovernanceContract {
  constructor(nexusGovernanceContractAddress, providerUrl) {
    this.nexusGovernanceContractAddress = nexusGovernanceContractAddress;
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.ethersProvider = new Ethers.providers.JsonRpcProvider(providerUrl);
  }

  async proposeProposal(proposalId, proposalData) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusGovernanceContractAddress);
    const tx = {
      from: this.nexusGovernanceContractAddress,
      to: this.nexusGovernanceContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${proposalId.toString(16)}${proposalData}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async voteOnProposal(proposalId, vote) {
    const txCount = await this.web3.eth.getTransactionCount(this.nexusGovernanceContractAddress);
    const tx = {
      from: this.nexusGovernanceContractAddress,
      to: this.nexusGovernanceContractAddress,
      value: '0',
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
      data: `0x${proposalId.toString(16)}${vote}`
    };
    const signedTx = await this.ethersProvider.signTransaction(tx, '0x1234567890abcdef');
    await this.ethersProvider.sendTransaction(signedTx);
  }

  async getProposal(proposalId) {
    return this.web3.eth.call({
      to: this.nexusGovernanceContractAddress,
      data: `0x${proposalId.toString(16)}`
    });
  }
}

module.exports = NexusGovernanceContract;
