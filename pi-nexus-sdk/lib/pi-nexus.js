import { Api } from './utils/api';
import { Wallet } from './utils/wallet';
import { Transaction } from './utils/transaction';
import { Contract } from './utils/contract';

class PiNexus {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
    this.api = new Api(apiUrl, apiKey);
    this.wallet = new Wallet(apiUrl, apiKey);
    this.transaction = new Transaction(apiUrl, apiKey);
    this.contract = new Contract(apiUrl, apiKey);
  }

  async getWallets() {
    return this.wallet.getWallets();
  }

  async getWallet(walletId) {
    return this.wallet.getWallet(walletId);
  }

  async createWallet(walletData) {
    return this.wallet.createWallet(walletData);
  }

  async getTransactions() {
    return this.transaction.getTransactions();
  }

  async getTransaction(transactionId) {
    return this.transaction.getTransaction(transactionId);
  }

  async createTransaction(transactionData) {
    return this.transaction.createTransaction(transactionData);
  }

  async getContracts() {
    return this.contract.getContracts();
  }

  async getContract(contractId) {
    return this.contract.getContract(contractId);
  }

  async createContract(contractData) {
    return this.contract.createContract(contractData);
  }
}

export default PiNexus;
