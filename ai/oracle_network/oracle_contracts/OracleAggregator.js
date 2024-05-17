const WeatherAPI = require('./weather-api/WeatherAPI')
const StockMarketAPI = require('./stock-market-api/StockMarketAPI')

class OracleAggregator {
  constructor () {
    this.weatherAPI = new WeatherAPI()
    this.stockMarketAPI = new StockMarketAPI('AAPL')
  }

  async updateAggregatedData (contract, api) {
    if (api === 'weather') {
      const data = await this.weatherAPI.getData()
      await contract.updateAggregatedData('weather', data)
    } else if (api === 'stock-market') {
      const data = await this.stockMarketAPI.getData()
      await contract.updateAggregatedData('stock-market', data)
    }
  }
}

module.exports = OracleAggregator
