const Web3 = require('web3');
const fs = require('fs');
const path = require('path');
const { PiTradeToken, TradeFinance } = require('../build/contracts');

const web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:8545'));

const migrate = async () => {
  const accounts = await web3.eth.getAccounts();
  const deployer = accounts[0];

  console.log(`Migrating contracts from account: ${deployer}`);

  const piTradeTokenAddress = fs.readFileSync(path.join(__dirname, '../build/contracts/PiTradeToken.address'), 'utf8');
  const tradeFinanceAddress = fs.readFileSync(path.join(__dirname, '../build/contracts/TradeFinance.address'), 'utf8');

  const piTradeToken = new web3.eth.Contract(PiTradeToken.abi, piTradeTokenAddress);
  const tradeFinance = new web3.eth.Contract(TradeFinance.abi, tradeFinanceAddress);

  const currentBlock = await web3.eth.getBlockNumber();
  console.log(`Current block number: ${currentBlock}`);

  const piTradeTokenBalance = await piTradeToken.methods.balanceOf(deployer).call();
  console.log(`PiTradeToken balance: ${piTradeTokenBalance}`);

  const tradeFinanceBalance = await tradeFinance.methods.getTradeBalance(deployer, deployer).call();
  console.log(`TradeFinance balance: ${tradeFinanceBalance}`);

  // Perform any necessary migrations here
  // ...

  console.log('Migration successful');
};

migrate()
  .then(() => console.log('Migration successful'))
  .catch((error) => console.error('Migration failed:', error));
