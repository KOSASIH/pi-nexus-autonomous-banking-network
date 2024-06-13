import { CCXT } from 'ccxt';

class CryptocurrencyExchange {
  constructor() {
    this.ccxt = new CCXT();
  }

  async getExchangeRates() {
    const rates = await this.ccxt.fetchTickers();
    return rates;
  }

  async executeTrade(symbol, amount) {
    const trade = await this.ccxt.createMarketOrder(symbol, amount);
    return trade;
  }
}

export default CryptocurrencyExchange;
