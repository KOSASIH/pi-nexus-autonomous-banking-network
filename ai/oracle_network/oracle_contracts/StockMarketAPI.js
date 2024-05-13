const axios = require('axios')

class StockMarketAPI {
  constructor (symbol) {
    this.symbol = symbol
  }

  async getData () {
    const response = await axios.get(
      `https://financialmodelingprep.com/api/v3/quote/${this.symbol}?apikey=YOUR_API_KEY`
    )
    return response.data[0].price
  }
}

module.exports = StockMarketAPI
