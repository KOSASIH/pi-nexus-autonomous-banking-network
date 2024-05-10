const { PRIVATE_KEY, PRIVATE_KEY_PASSWORD, PRIVATE_KEY_PASSWORD_SALT, ENDPOINT, CHAIN_ID } = process.env;

const Web3 = require("web3");
const EthereumTx = require("ethereumjs-tx");

const web3 = new Web3(new Web3.providers.HttpProvider(ENDPOINT));

const createTransaction = async (from, to, value) => {
  const nonce = await web3.eth.getTransactionCount(from);
  const gasPrice = await web3.eth.getGasPrice();
  const gasLimit = 21000;

  const tx = new EthereumTx({
    nonce,
    gasPrice,
    gasLimit,
    to,
    value,
  });

  const privateKey = web3.utils.sha3(PRIVATE_KEY_PASSWORD + PRIVATE_KEY_PASSWORD_SALT);
  tx.sign(privateKey);

  const serializedTx = tx.serialize();
  const transactionHash = await web3.eth.sendSignedTransaction("0x" + serializedTx.toString("hex"));

  return transactionHash;
};

module.exports = createTransaction;
