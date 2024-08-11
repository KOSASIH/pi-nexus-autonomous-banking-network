import Web3 from 'web3';
import { NexusToken } from '../contracts/NexusToken.sol';
import { BlockchainService } from './BlockchainService';

class NexusService {
  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    this.nexusToken = new NexusToken('0x...NexusTokenAddress...');
    this.blockchainService = new BlockchainService();
  }

  async getBalance(address) {
    try {
      const balance = await this.nexusToken.methods.balanceOf(address).call();
      return balance;
    } catch (error) {
      throw new Error(`Failed to get balance: ${error.message}`);
    }
  }

  async transferTokens(sender, recipient, amount) {
    try {
      const txCount = await this.web3.eth.getTransactionCount(sender);
      const tx = {
        from: sender,
        to: this.nexusToken.address,
        value: '0',
        gas: '20000',
        gasPrice: '20',
        nonce: txCount,
        data: this.nexusToken.methods.transfer(recipient, amount).encodeABI(),
      };
      const signedTx = await this.web3.eth.accounts.signTransaction(tx, '0x...privateKey...');
      const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
      return receipt;
    } catch (error) {
      throw new Error(`Failed to transfer tokens: ${error.message}`);
    }
  }
}

export default NexusService;
