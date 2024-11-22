const Web3 = require('web3');
const PiToken = require('../build/PiToken.json');

async function deploy() {
    const web3 = new Web3('https://your.ethereum.node');
    const accounts = await web3.eth.getAccounts();
    const token = await new web3.eth.Contract(PiToken.abi)
        .deploy({ data: PiToken.evm.bytecode.object, arguments: [1000000] })
        .send({ from: accounts[0], gas: '1000000' });
    console.log(`Token deployed at: ${token.options.address}`);
}

deploy();
