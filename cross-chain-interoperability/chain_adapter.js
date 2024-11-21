const axios = require('axios')

class ChainAdapter {
  constructor (config) {
    this.config = config
  }

  async getBalance (address) {
    try {
      const response = await axios.get(
        `${this.config.chainApiUrl}/accounts/${address}`,
        {
          headers: {
            Authorization: `Bearer ${this.config.apiKey}`
          }
        }
      )
      return response.data.balance
    } catch (error) {
      console.error(
        `Error getting balance for address ${address}:`,
        error.message
      )
      throw error
    }
  }

  async transfer (fromAddress, toAddress, amount) {
    try {
      const response = await axios.post(
        `${this.config.chainApiUrl}/transfers`,
        {
          from: fromAddress,
          to: toAddress,
          amount
        },
        {
          headers: {
            Authorization: `Bearer ${this.config.apiKey}`
          }
        }
      )
      return response.data
    } catch (error) {
      console.error(
        `Error transferring funds from ${fromAddress} to ${toAddress}:`,
        error.message
      )
      throw error
    }
  }
}

module.exports = ChainAdapter
