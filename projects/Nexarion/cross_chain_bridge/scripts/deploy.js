const Web3 = require('web3');
const { Ethereum } = require('@ethereumjs/ethereumjs-vm');
const { BinanceSmartChain } = require('@binance-chain/bsc-bridge-solidity');

const web3Ethereum = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
const web3BinanceSmartChain = new Web3(new Web3.providers.HttpProvider('https://bsc-dataseed.binance.org/api/v1/bc/BSC/main'));

const CrossChainBridge = require('../contracts/CrossChainBridge.sol');
const EthereumBridge = require('../contracts/EthereumBridge.sol');
const BinanceSmartChainBridge = require('../contracts/BinanceSmartChainBridge.sol');

async function deploy() {
  const accounts = await web3Ethereum.eth.getAccounts();
  const owner = accounts[0];

  const crossChainBridge = new web3Ethereum.eth.Contract(CrossChainBridge.abi);
  const ethereumBridge = new web3Ethereum.eth.Contract(EthereumBridge.abi);
  const binanceSmartChainBridge = new web3BinanceSmartChain.eth.Contract(BinanceSmartChainBridge.abi);

  await crossChainBridge.deploy({ data: CrossChainBridge.bytecode })
    .send({ from: owner, gas: 2000000 })
    .on('transactionHash', hash => console.log(`EthereumBridge deployment transaction hash: ${hash}`))
    .on('confirmation', (confirmationNumber, receipt) => {
      console.log(`EthereumBridge deployment confirmation number: ${confirmationNumber}`);
      console.log(`EthereumBridge deployment receipt: ${receipt}`);
    });

  await binanceSmartChainBridge.deploy({ data: BinanceSmartChainBridge.bytecode })
    .send({ from: owner, gas: 2000000 })
    .on('transactionHash', hash => console.log(`BinanceSmartChainBridge deployment transaction hash: ${hash}`))
    .on('confirmation', (confirmationNumber, receipt) => {
      console.log(`BinanceSmartChainBridge deployment confirmation number: ${confirmationNumber}`);
      console.log(`BinanceSmartChainBridge deployment receipt: ${receipt}`);
    });

  console.log('Deployment complete!');
}

deploy();
