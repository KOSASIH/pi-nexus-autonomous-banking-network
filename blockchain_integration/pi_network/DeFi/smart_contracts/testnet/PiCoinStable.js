// Import the Stellar SDK
const StellarSdk = require('stellar-sdk');

// Connect to the Stellar test network
const server = new StellarSdk.Server('https://horizon-testnet.stellar.org');

// Replace with your secret key
const issuerSecret = 'YOUR_ISSUER_SECRET_KEY'; // Replace with your actual secret key
const issuerKeypair = StellarSdk.Keypair.fromSecret(issuerSecret);
const assetName = 'PI'; // Token name
const assetAmount = '100000000000'; // Total supply (100 billion tokens)

// Pegged value in USD
const peggedValue = 314159; // Pegged value in USD

async function createToken() {
    try {
        // Create a new asset
        const asset = new StellarSdk.Asset(assetName, issuerKeypair.publicKey());

        // Load the issuer account
        const account = await server.loadAccount(issuerKeypair.publicKey());

        // Create a transaction to issue the asset
        const transaction = new StellarSdk.TransactionBuilder(account, {
            fee: await server.fetchBaseFee(),
            networkPassphrase: StellarSdk.Networks.TESTNET,
        })
        .addOperation(StellarSdk.Operation.changeTrust({
            asset: asset,
            limit: assetAmount,
        }))
        .setTimeout(30)
        .build();

        // Sign the transaction
        transaction.sign(issuerKeypair);

        // Submit the transaction
        const result = await server.submitTransaction(transaction);
        console.log('Token created successfully:', result);
        console.log(`The token ${assetName} is pegged to a value of $${peggedValue}.`);

        // Check the issuer's balance
        await checkIssuerBalance();
    } catch (error) {
        console.error('Error creating token:', error);
    }
}

async function checkIssuerBalance() {
    try {
        const account = await server.loadAccount(issuerKeypair.publicKey());
        console.log(`Issuer balance: ${account.balances}`);
    } catch (error) {
        console.error('Error fetching issuer balance:', error);
    }
}

// Execute the function to create the token
createToken();
