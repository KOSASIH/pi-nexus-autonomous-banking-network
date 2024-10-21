const Web3 = require('web3');
const web3 = new Web3(
  new Web3.providers.HttpProvider(
    'https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
  ),
);

const transactionProcessor = async (req, res) => {
  const transaction = req.body.transaction;
  const txCount = await web3.eth.getTransactionCount(transaction.from);
  const tx = {
    from: transaction.from,
    to: transaction.to,
    value: web3.utils.toWei(transaction.amount, 'ether'),
    gas: '21000',
    gasPrice: '10000000000',
    nonce: web3.utils.toHex(txCount),
  };
  const signedTx = await web3.eth.accounts.signTransaction(
    tx,
    transaction.privateKey,
  );
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  res.json(receipt);
};

module.exports = transactionProcessor;
