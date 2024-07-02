import web3 from './blockchain';

const transaction = {
  async sendTransaction(from, to, value) {
    const txCount = await web3.eth.getTransactionCount(from);
    const gasPrice = await web3.eth.getGasPrice();
    const gasEstimate = await web3.eth.estimateGas({
      from,
      to,
      value,
    });

    const tx = {
      from,
      to,
      value,
      gas: gasEstimate,
      gasPrice,
      nonce: txCount,
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...');

    await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  },
};

export default transaction;
