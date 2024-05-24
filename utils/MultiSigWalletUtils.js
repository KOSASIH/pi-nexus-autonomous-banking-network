const Web3 = require("web3");

class MultiSigWalletUtils {
  // The function to generate a new multi-signature wallet address
  static generateAddress(owners) {
    const web3 = new Web3();
    const hash = web3.utils.soliditySha3(owners);
    return web3.utils.keccak256(hash).slice(26);
  }

  // The function to generate a new signature for a transaction
  static generateSignature(privateKey, transaction) {
    const web3 = new Web3();
    const hash = web3.utils.soliditySha3(
      transaction.to,
      transaction.value,
      transaction.data,
      transaction.requiredSignatures,
      transaction.signatures,
    );
    const signature = web3.eth.accounts.sign(hash, privateKey);
    return signature.signature;
  }
}

module.exports = MultiSigWalletUtils;
