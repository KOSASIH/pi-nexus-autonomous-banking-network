import web3 from './blockchain';

const mining = {
  async mineBlock() {
    const blockNumber = await web3.eth.getBlockNumber();
    const block = await web3.eth.getBlock(blockNumber);

    const nonce = await web3.eth.getTransactionCount('0x...');
    const gasPrice = await web3.eth.getGasPrice();
    const gasEstimate = await web3.eth.estimateGas({
      from: '0x...',
      to: '0x...',
      value: 1,
    });

    const tx = {
      from: '0x...',
      to: '0x...',
      value: 1,
      gas: gasEstimate,
      gasPrice,
      nonce,
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...');

    await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
  },
};

export default mining;
