import axios from 'axios';

class ExchangeModel {
  constructor(exchange) {
    this.exchange = exchange;
    this.apiUrl = exchange.apiUrl;
    this.apiKey = exchange.apiKey;
    this.apiSecret = exchange.apiSecret;
  }

  async getTicker(symbol) {
    const response = await axios.get(`${this.apiUrl}/ticker`, {
      params: { symbol },
      headers: {
        'X-MBX-APIKEY': this.apiKey,
        'X-MBX-SECRET-KEY': this.apiSecret
      }
    });
    return response.data;
  }

  async placeOrder(symbol, side, quantity, price) {
    const response = await axios.post(`${this.apiUrl}/order`, {
      symbol,
      side,
      quantity,
      price,
      type: 'limit'
    }, {
      headers: {
        'X-MBX-APIKEY': this.apiKey,
        'X-MBX-SECRET-KEY': this.apiSecret
      }
    });
    return response.data;
  }
}

export default ExchangeModel;
