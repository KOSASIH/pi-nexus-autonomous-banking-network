import StellarSdk from 'stellar-sdk';

class StellarService {
    constructor() {
        StellarSdk.Network.useTestNetwork(); // Change to mainnet for production
        this.server = new StellarSdk.Server('https://horizon-testnet.stellar.org');
    }

    async transferAsset(sourceSecret, destinationPublicKey, amount, assetCode) {
        const sourceKeypair = StellarSdk.Keypair.fromSecret(sourceSecret);
        const sourceAccount = await this.server.loadAccount(sourceKeypair.publicKey());

        const transaction = new StellarSdk.TransactionBuilder(sourceAccount, {
            fee: StellarSdk.BASE_FEE,
            networkPassphrase: StellarSdk.Networks.TESTNET,
        })
        .addOperation(StellarSdk.Operation.payment({
            destination: destinationPublicKey,
            asset: new StellarSdk.Asset(assetCode, 'IssuerPublicKey'), // Replace with actual issuer
            amount: amount.toString(),
        }))
        .setTimeout(30)
        .build();

        transaction.sign(sourceKeypair);
        return await this.server.submitTransaction(transaction);
    }
}

export default new StellarService();
