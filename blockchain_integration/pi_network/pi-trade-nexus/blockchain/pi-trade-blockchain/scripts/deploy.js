const Web3 = require('web3');
const fs = require('fs');
const path = require('path');
const { PiTradeToken, TradeFinance } = require('../build/contracts');

const web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:8545'));

const deploy = async () => {
  const accounts = await web3.eth.getAccounts();
  const deployer = accounts[0];

  console.log(`Deploying contracts from account: ${deployer}`);

  const piTradeToken = await PiTradeToken.new(web3.utils.toWei('100000000', 'ether'), {
    from: deployer,
    gas: '2000000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });

  console.log(`PiTradeToken deployed at: ${piTradeToken.address}`);

  const tradeFinance = await TradeFinance.new(piTradeToken.address, {
    from: deployer,
    gas: '2000000',
    gasPrice: web3.utils.toWei('20', 'gwei'),
  });

  console.log(`TradeFinance deployed at: ${tradeFinance.address}`);

  fs.writeFileSync(path.join(__dirname, '../build/contracts/PiTradeToken.json'), JSON.stringify(piTradeToken.abi, null, 2));
  fs.writeFileSync(path.join(__dirname, '../build/contracts/TradeFinance.json'), JSON.stringify(tradeFinance.abi, null, 2));
};

deploy()
  .then(() => console.log('Deployment successful'))
  .catch((error) => console.error('Deployment failed:', error));
