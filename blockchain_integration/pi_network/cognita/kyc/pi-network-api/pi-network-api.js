const StellarSdk = require('stellar-sdk');

const piNetworkApi = {
  loadAccount: async (publicKey) => {
    const server = new StellarSdk.Server('https://api.testnet.minepi.com');
    const account = await server.loadAccount(publicKey);
    return account;
  },

  fetchBaseFee: async () => {
    const server = new StellarSdk.Server('https://api.testnet.minepi.com');
    const baseFee = await server.fetchBaseFee();
    return baseFee;
  },

  createPayment: async (recipientAddress, amount, memo) => {
    const server = new StellarSdk.Server('https://api.testnet.minepi.com');
    const payment = StellarSdk.Operation.payment({
      destination: recipientAddress,
      asset: StellarSdk.Asset.native(),
      amount: amount.toString(),
    });
    const timebounds = await server.fetchTimebounds(180);
    const transaction = new StellarSdk.TransactionBuilder(myAccount, {
      fee: baseFee,
      networkPassphrase: 'Pi Testnet',
      timebounds: timebounds,
    })
     .addOperation(payment)
     .addMemo(StellarSdk.Memo.text(memo));
    return transaction;
  },

  submitTransaction: async (transaction) => {
    const server = new StellarSdk.Server('https://api.testnet.minepi.com');
    const response = await server.submitTransaction(transaction);
    return response;
  },
};

module.exports = piNetworkApi;
