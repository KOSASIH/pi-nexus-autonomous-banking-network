import Web3 from 'web3';
import { PiNetworkRouter } from './PiNetworkRouter';

const web3 = new Web3(
  new Web3.providers.HttpProvider(
    'https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
  ),
);

const piNetworkRouterAddress = '0x...'; // Replace with the deployed PiNetworkRouter address
const piNetworkRouter = new PiNetworkRouter(piNetworkRouterAddress, web3);

async function main() {
  const fromAddress = '0x...'; // Replace with the sender's address
  const toAddress = '0x...'; // Replace with the recipient's address
  const amount = 1; // Replace with the amount to transfer

  const txCount = await web3.eth.getTransactionCount(fromAddress);
  const tx = {
    from: fromAddress,
    to: piNetworkRouterAddress,
    value: web3.utils.toWei(amount, 'ether'),
    gas: '20000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
    nonce: txCount,
  };

  const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's private key
  const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

  console.log(`Transaction receipt: ${receipt.transactionHash}`);
}

main();
