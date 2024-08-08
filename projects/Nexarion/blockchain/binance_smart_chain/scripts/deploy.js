const Web3 = require('web3');
const { BinanceSmartChain } = require('@binance-chain/bsc-bridge-solidity');

const web3 = new Web3(new Web3.providers.HttpProvider('https://bsc-dataseed.binance.org/api/v1/bc/BSC/main'));

const BEP20Token = require('../contracts/BEP20Token.sol');
const BinanceSmartChainRouter = require('../contracts/BinanceSmartChainRouter.sol');

async function deploy() {
  const accounts = await web3.eth.getAccounts();
  const owner = accounts[0];

  const bep20Token = new web3.eth.Contract(BEP20Token.abi);
  const binanceSmartChainRouter = new web3.eth.Contract(BinanceSmartChainRouter.abi);

  await bep20Token.deploy({ data: BEP20Token.bytecode })
    .send({ from: owner, gas: 2000000 })
    .on('transactionHash', hash => console.log(`BEP20Token deployment transaction hash: ${hash}`))
    .on('confirmation', (confirmationNumber, receipt) => {
      console.log(`BEP20Token deployment confirmation number: ${confirmationNumber}`);
      console.log(`BEP20Token deployment receipt: ${receipt}`);
    });

  await binanceSmartChainRouter.deploy({ data: BinanceSmartChainRouter.bytecode })
    .send({ from: owner, gas: 2000000 })
    .on('transactionHash', hash => console.log(`BinanceSmartChainRouter deployment transaction hash: ${hash}`))
    .on('confirmation', (confirmationNumber, receipt) => {
      console.log(`BinanceSmartChainRouter deployment confirmation number: ${confirmationNumber}`);
      console.log(`BinanceSmartChainRouter deployment receipt: ${receipt}`);
    });
}

deploy();
