const Web3 = require('web3');
const MultiSigWallet = require('../build/MultiSigWallet.json');

async function deploy() {
    const web3 = new Web3('https://your.ethereum.node');
    const accounts = await web3.eth.getAccounts();
    const owners = [accounts[0], accounts[1], accounts[2]]; // Replace with your owner addresses
    const required = 2; // Replace with your required number of owners
    const wallet = await new web3.eth.Contract(MultiSigWallet.abi)
        .deploy({ data: MultiSigWallet.evm.bytecode.object, arguments: [owners, required] })
        .send({ from: accounts[0], gas: '1000000' });
    console.log(`Multi-sig wallet deployed at: ${wallet.options.address}`);
}

deploy();
