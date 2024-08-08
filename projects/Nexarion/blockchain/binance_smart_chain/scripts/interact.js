const Web3 = require('web3');
const BEP20Token = require('../contracts/BEP20Token.sol');
const BinanceSmartChainRouter = require('../contracts/BinanceSmartChainRouter.sol');

const web3 = new Web3(new Web3.providers.HttpProvider('https://bsc-dataseed.binance.org/api/v1/bc/BSC/main'));

const bep20TokenAddress = '0x...'; // BEP20Token contract address
const binanceSmartChainRouterAddress = '0x...'; // BinanceSmartChainRouter contract address

const bep20Token = new web3.eth.Contract(BEP20Token.abi, bep20TokenAddress);
const binanceSmartChainRouter = new web3.eth.Contract(BinanceSmartChainRouter.abi, binanceSmartChainRouterAddress);

async function interact() {
  const accounts = await web3.eth.getAccounts();
  const owner = accounts[0];

  // Transfer 10 tokens from the owner to another account
  await bep20Token.methods.transfer('0x...', 10).send({ from: owner, gas: 2000000 });

  // Get the balance of the owner
  const balance = await bep20Token.methods.balanceOf(owner).call();
  console.log(`Owner balance: ${balance}`);

  // Transfer 1 BNB from the owner to another account using the Binance Smart Chain Router
  await binanceSmartChainRouter.methods.transferBnb('0x...', 1).send({ from: owner, gas: 2000000 });
}

interact();
