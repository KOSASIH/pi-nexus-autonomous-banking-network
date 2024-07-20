// blockchain-module/interact.js
const Web3 = require('web3');
const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const contractAddress = '0x...';
const contractABI = [...];

const contract = new web3.eth.Contract(contractABI, contractAddress);

async function deposit(amount) {
    const txCount = await web3.eth.getTransactionCount();
    const tx = {
        from: '0x...',
        to: contractAddress,
        value: web3.utils.toWei(amount, 'ether'),
        gas: '20000',
        gasPrice: web3.utils.toWei('20', 'gwei'),
        nonce: txCount
    };
    const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...');
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    console.log(`Transaction receipt: ${receipt.transactionHash}`);
}

async function withdraw(amount) {
    //...
}

async function getBalance(account) {
    //...
}
