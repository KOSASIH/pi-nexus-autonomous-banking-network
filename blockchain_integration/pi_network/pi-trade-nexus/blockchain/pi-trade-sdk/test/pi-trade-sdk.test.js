import { PiTradeSDK } from '../lib/pi-trade-sdk';
import Web3 from 'web3';
import { accounts, contract } from '@openzeppelin/test-environment';

describe('PiTradeSDK', () => {
  let piTradeSDK;
  let piTradeTokenAddress;
  let tradeFinanceAddress;
  let providerUrl;

  beforeEach(async () => {
    providerUrl = 'http://localhost:8545';
    piTradeTokenAddress = '0x1234567890abcdef';
    tradeFinanceAddress = '0xfedcba9876543210';
    piTradeSDK = new PiTradeSDK(providerUrl, piTradeTokenAddress, tradeFinanceAddress);
  });

  it('should get account balance', async () => {
    const accountAddress = accounts[0];
    const balance = await piTradeSDK.getAccountBalance(accountAddress);
    expect(balance).to.be.a('string');
  });

  it('should get PiTradeToken balance', async () => {
    const accountAddress = accounts[0];
    const balance = await piTradeSDK.getPiTradeTokenBalance(accountAddress);
    expect(balance).to.be.a('string');
  });

  it('should get TradeFinance balance', async () => {
    const accountAddress = accounts[0];
    const balance = await piTradeSDK.getTradeFinanceBalance(accountAddress);
    expect(balance).to.be.a('string');
  });

  it('should transfer PiTradeToken', async () => {
    const fromAccount = accounts[0];
    const toAccount = accounts[1];
    const amount = Web3.utils.toWei('10', 'ether');
    const tx = await piTradeSDK.transferPiTradeToken(fromAccount, toAccount, amount);
    expect(tx).to.be.a('object');
  });

  it('should initiate trade', async () => {
    const fromAccount = accounts[0];
    const toAccount = accounts[1];
    const amount = Web3.utils.toWei('10', 'ether');
    const tx = await piTradeSDK.initiateTrade(fromAccount, toAccount, amount);
    expect(tx).to.be.a('object');
  });

  it('should confirm trade', async () => {
    const fromAccount = accounts[0];
    const toAccount = accounts[1];
    const amount = Web3.utils.toWei('10', 'ether');
    const tx = await piTradeSDK.confirmTrade(fromAccount, toAccount, amount);
    expect(tx).to.be.a('object');
  });

  it('should cancel trade', async () => {
    const fromAccount = accounts[0];
    const toAccount = accounts[1];
    const tx = await piTradeSDK.cancelTrade(fromAccount, toAccount);
    expect(tx).to.be.a('object');
  });
});
