import { ethers } from 'ethers.js';
import { KEVM } from 'kevm';
import { JAAK } from 'jaak';
import { elliptic } from 'elliptic';
import { scrypt } from 'scrypt-js';

class PiWallet {
  constructor() {
    this.ethersProvider = new ethers.providers.JsonRpcProvider();
    this.kevm = new KEVM();
    this.jaak = new JAAK();
    this.ecdsa = new elliptic.ec('secp256k1');
    this.scryptParams = {
      N: 16384,
      r: 8,
      p: 1,
    };
  }

  async createWallet() {
    const wallet = await this.ethersProvider.createWallet();
    return wallet;
  }

  async getBalance(address) {
    const balance = await this.ethersProvider.getBalance(address);
    return balance;
  }

  async signTransaction(transaction) {
    const signature = await this.ecdsa.sign(transaction, this.scryptParams);
    return signature;
  }

  async deploySmartContract(contractCode) {
    const compiledContract = await this.kevm.compile(contractCode);
    const deployedContract = await this.jaak.deployContract(compiledContract);
    return deployedContract;
  }

  async executeSmartContract(contractAddress, functionName, functionArgs) {
    const contractInstance = await this.jaak.getContractInstance(contractAddress);
    const functionResult = await contractInstance[functionName](...functionArgs);
    return functionResult;
  }

  async updateWalletEndpoint() {
    // Use the PATCH method to update the wallet endpoint
    const walletEndpoint = await this.ethersProvider.getWalletEndpoint();
    const updatedWalletEndpoint = await this.ethersProvider.updateWalletEndpoint(walletEndpoint);
    return updatedWalletEndpoint;
  }

  async addOrUpdateWalletEndpoint() {
    // Use the PUT method to add or update the wallet endpoint
    const walletEndpoint = await this.ethersProvider.getWalletEndpoint();
    const updatedWalletEndpoint = await this.ethersProvider.addOrUpdateWalletEndpoint(walletEndpoint);
    return updatedWalletEndpoint;
  }
}

export { PiWallet as Wallet };
