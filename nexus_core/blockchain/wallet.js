import web3 from './blockchain';

const wallet = {
  async createAccount() {
    const account = web3.eth.accounts.create();
    return account;
  },

  async getAccountBalance(account) {
    const balance = await web3.eth.getBalance(account.address);
    return balance;
  },
};

export default wallet;
