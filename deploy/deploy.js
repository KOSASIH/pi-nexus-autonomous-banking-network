const HDWalletProvider = require('@truffle/hdwallet-provider');
const Web3 = require('web3');
const { interface, bytecode } = require('./build/contracts/PiNexusIdentityManager.json');

const mnemonic = process.env.MNEMONIC;
const provider = new HDWalletProvider(mnemonic, 'http://localhost:7545');
const web3 = new Web3(provider);

const deploy = async () => {
  const accounts = await web3.eth.getAccounts();

  console.log('Deploying PiNexusIdentityManager...');
  const contract = new web3.eth.Contract(JSON.parse(interface));
  const transaction = contract.deploy({ data: bytecode });
  const options = { from: accounts[0], gas: 4700000 };
  const receipt = await transaction.send(options);
  console.log(`PiNexusIdentityManager deployed at ${receipt.contractAddress}`);
};

deploy();
