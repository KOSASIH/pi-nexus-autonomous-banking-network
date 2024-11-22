const Web3 = require('web3');
const Governance = require('../build/Governance.json');
const PiToken = require('../build/PiToken.json');

async function deploy() {
    const web3 = new Web3('https://your.ethereum.node');
    const accounts = await web3.eth.getAccounts();
    const tokenAddress = '0x...'; // Replace with your token contract address
    const governance = await new web3.eth.Contract(Governance.abi)
        .deploy({ data: Governance.evm.bytecode.object, arguments: [tokenAddress] })
        .send({ from: accounts[0], gas: '1000000' });
    console.log(`Governance contract deployed at: ${governance.options.address}`);
}

deploy();
