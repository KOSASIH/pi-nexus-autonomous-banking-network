import StellarSDK from 'stellar-sdk';

const stellar = new StellarSDK.Server('https://horizon-testnet.stellar.org');

export async function getStellarBalance(address) {
  const account = await stellar.accounts().accountId(address).call();
  return account.balances[0].balance;
}

export async function sendStellarPayment(fromAddress, toAddress, amount) {
  const sourceKeys = StellarSDK.Keypair.fromSecret(fromAddress);
  const destinationId = toAddress;

  const transaction = new StellarSDK.TransactionBuilder(new StellarSDK.Account(fromAddress, '1'))
    .addOperation(StellarSDK.Operation.payment({
      destination: destinationId,
      asset: StellarSDK.Asset.native(),
      amount: amount
    }))
    .build();

  transaction.sign(sourceKeys);

  try {
    const result = await stellar.submitTransaction(transaction);
    return result;
  } catch (error) {
    throw error;
  }
}

export async function createStellarAccount(address) {
  const sourceKeys = StellarSDK.Keypair.fromSecret(address);
  const newAccount = StellarSDK.Keypair.random();

  const transaction = new StellarSDK.TransactionBuilder(new StellarSDK.Account(address, '1'))
    .addOperation(StellarSDK.Operation.createAccount({
      destination: newAccount.publicKey(),
      startingBalance: '1'
    }))
    .build();

  transaction.sign(sourceKeys);

  try {
    const result = await stellar.submitTransaction(transaction);
    return result;
  } catch (error) {
    throw error;
  }
}
