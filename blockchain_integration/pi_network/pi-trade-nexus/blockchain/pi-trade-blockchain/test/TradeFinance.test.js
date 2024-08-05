const Web3 = require('web3');
const { TradeFinance } = require('../build/contracts');
const { expect } = require('chai');

describe('TradeFinance', () => {
  let web3;
  let tradeFinance;
  let piTradeToken;
  let accounts;

  beforeEach(async () => {
    web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:8545'));
    accounts = await web3.eth.getAccounts();
    piTradeToken = await PiTradeToken.new(web3.utils.toWei('100000000', 'ether'), {
      from: accounts[0],
      gas: '2000000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });
    tradeFinance = await TradeFinance.new(piTradeToken.address, {
      from: accounts[0],
      gas: '2000000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });
  });

  it('should allow trade initiation', async () => {
    const buyer = accounts[0];
    const seller = accounts[1];
    const amount = web3.utils.toWei('100', 'ether');

    await tradeFinance.methods.initiateTrade(seller, amount).send({
      from: buyer,
      gas: '2000000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const tradeBalance = await tradeFinance.methods.getTradeBalance(buyer, seller).call();
    expect(tradeBalance).to.equal(amount);
  });

  it('should allow trade confirmation', async () => {
    const buyer = accounts[0];
    const seller = accounts[1];
    const amount = web3.utils.toWei('100', 'ether');

    await tradeFinance.methods.initiateTrade(seller, amount).send({
      from: buyer,
      gas: '2000000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    await tradeFinance.methods.confirmTrade(buyer, amount).send({
      from: seller,
      gas: '2000000',
      gasPrice: web3.utils.toWei('20', 'gwei'),
    });

    const tradeBalance = await tradeFinance.methods.getTrade
