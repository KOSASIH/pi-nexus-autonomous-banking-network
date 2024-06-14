import Web3 from 'web3';
import { abi, bytecode } from '../smart-contracts/PiGenesisToken.sol';

class BlockchainService {
  async getBalance(address) {
    const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    const balance = await web3.eth.getBalance(address);
    return balance;
  }

  async getTransactionCount(address) {
    const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    const transactionCount = await web3.eth.getTransactionCount(address);
    return transactionCount;
  }

  async sendTransaction(from, to, value) {
    const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    const txCount = await web3.eth.getTransactionCount(from);
    const tx = {
      from,
      to,
      value,
      gas: '20000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
      nonce: txCount,
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    return receipt;
  }

  async deployContract() {
    const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    const contract = new web3.eth.Contract(abi);
    const deployTx = contract.deploy({ data: bytecode, arguments: [] });
    const receipt = await web3.eth.sendTransaction(deployTx);
    return receipt;
  }
}

export default BlockchainService;
