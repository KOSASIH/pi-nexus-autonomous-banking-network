import Web3 from 'web3';
import { PiTradeToken } from './contracts/PiTradeToken';
import { TradeFinance } from './contracts/TradeFinance';
import { utils } from './utils';

class PiTradeSDK {
  constructor(providerUrl, piTradeTokenAddress, tradeFinanceAddress) {
    this.web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
    this.piTradeToken = new PiTradeToken(piTradeTokenAddress);
    this.tradeFinance = new TradeFinance(tradeFinanceAddress);
  }

  async getAccountBalance(accountAddress) {
    return this.web3.eth.getBalance(accountAddress);
  }

  async getPiTradeTokenBalance(accountAddress) {
    return this.piTradeToken.methods.balanceOf(accountAddress).call();
  }

  async getTradeFinanceBalance(accountAddress) {
    return this.tradeFinance.methods.getTradeBalance(accountAddress).call();
  }

  async transferPiTradeToken(fromAccount, toAccount, amount) {
    return this.piTradeToken.methods.transfer(toAccount, amount).send({
      from: fromAccount,
      gas: '2000000',
      gasPrice: this.web3.utils.toWei('20', 'gwei'),
    });
  }

  async initiateTrade(fromAccount, toAccount, amount) {
    return this.tradeFinance.methods.initiateTrade(toAccount, amount).send({
      from: fromAccount,
      gas: '2000000',
      gasPrice: this.web3.utils.toWei('20', 'gwei'),
    });
  }

  async confirmTrade(fromAccount, toAccount, amount) {
    return this.tradeFinance.methods.confirmTrade(fromAccount, amount).send({
      from: toAccount,
      gas: '2000000',
      gasPrice: this.web3.utils.toWei('20', 'gwei'),
    });
  }

  async cancelTrade(fromAccount, toAccount) {
    return this.tradeFinance.methods.cancelTrade(toAccount).send({
      from: fromAccount,
      gas: '2000000',
      gasPrice: this.web3.utils.toWei('20', 'gwei'),
    });
  }
}

export default PiTradeSDK;
