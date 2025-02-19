const StellarSdk = require('stellar-sdk');
const server = new StellarSdk.Server('https://horizon-testnet.stellar.org');

class StellarService {
    constructor() {
        StellarSdk.Network.useTestNetwork();
    }

    async createAccount() {
        const pair = StellarSdk.Keypair.random();
        // Fund the account using the faucet
        await fundAccount(pair.publicKey());
        return pair;
    }

    async sendPayment(sourceSecret, destinationPublicKey, amount) {
        const sourceKeypair = StellarSdk.Keypair.fromSecret(sourceSecret);
        const sourceAccount = await server.loadAccount(sourceKeypair.publicKey());

        const transaction = new StellarSdk.TransactionBuilder(sourceAccount, {
            fee: StellarSdk.BASE_FEE,
            networkPassphrase: StellarSdk.Networks.TESTNET,
        })
        .addOperation(StellarSdk.Operation.payment({
            destination: destinationPublicKey,
            asset: StellarSdk.Asset.native(), // XLM
            amount: amount.toString(),
        }))
        .setTimeout(30)
        .build();

        transaction.sign(sourceKeypair);

        try {
            const result = await server.submitTransaction(transaction);
            return result;
        } catch (error) {
            throw new Error('Transaction failed: ' + error.message);
        }
    }
}

module.exports = new StellarService();
