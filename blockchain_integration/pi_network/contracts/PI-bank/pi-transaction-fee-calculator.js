const ethers = require('ethers');
const Avalanche = require('avalanche').Avalanche;
require('dotenv').config();

const privateKey = process.env.PRIVATEKEY;
const nodeURL = "https://api.avax-test.network/ext/bc/C/rpc";
const HTTPSProvider = new ethers.providers.JsonRpcProvider(nodeURL);
const chainId = 43113;
const avalanche = new Avalanche(
  "api.avax-test.network",
  undefined,
"https",
  chainId
);
const cchain = avalanche.CChain();
const wallet = new ethers.Wallet(privateKey);
const address = wallet.address;

async function calcFeeData() {
  const baseFee = await cchain.getBaseFee(1);
  const maxFee = baseFee.mul(125).div(100);
  const maxPriorityFee = baseFee.mul(50).div(100);
  return {
    baseFee: ethers.utils.formatUnits(baseFee, 'gwei'),
    maxFee: ethers.utils.formatUnits(maxFee, 'gwei'),
    maxPriorityFee: ethers.utils.formatUnits(maxPriorityFee, 'gwei')
  };
}

async function sendAvax(amount, to, maxFee, maxPriorityFee, nonce) {
  const feeData = await calcFeeData();
  const tx = {
    to: to,
    value: ethers.utils.parseEther(amount),
    gasLimit: 21000,
    maxFeePerGas: ethers.utils.parseUnits(maxFee, 'gwei'),
    maxPriorityFeePerGas: ethers.utils.parseUnits(maxPriorityFee, 'gwei'),
    nonce: nonce
  };
  const signedTx = await wallet.sign(tx);
  const txHash = await HTTPSProvider.sendTransaction(signedTx);
  console.log(`Transaction sent with hash: ${txHash}`);
  console.log(`View transaction with nonce ${nonce}: https://testnet.snowtrace.io/tx/${txHash}`);
}

// Example usage:
// sendAvax("0.01", "0x856EA4B78947c3A5CD2256F85B2B147fEBDb7124", 100, 10, 25);

module.exports = {
  sendAvax
};
