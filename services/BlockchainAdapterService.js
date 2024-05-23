const Bitcoin = require('bitcoinjs-lib')
const Litecoin = require('litecoin-js')

class BlockchainAdapterService {
  // The constructor
  constructor (piNetworkService, bitcoinService, litecoinService) {
    this.piNetworkService = piNetworkService
    this.bitcoinService = bitcoinService
    this.litecoinService = litecoinService
  }

  // The function to transfer assets to the Bitcoin network
  async transferToBitcoin (piAddress, bitcoinAddress, amount) {
    // Get the PI Network balance
    const piBalance = await this.piNetworkService.getBalance(piAddress)

    // Check if there is enough balance
    if (piBalance < amount) {
      throw new Error('Insufficient balance')
    }

    // Convert the PI Network amount to Bitcoin
    const bitcoinAmount = this.piNetworkService.convertToBitcoin(amount)

    // Generate a Bitcoin transaction
    const transaction = this.bitcoinService.createTransaction(
      bitcoinAddress,
      bitcoinAmount
    )

    // Sign the transaction with the private key
    const privateKey = this.bitcoinService.getPrivateKey(piAddress)
    this.bitcoinService.signTransaction(transaction, privateKey)

    // Broadcast the transaction to the Bitcoin network
    return this.bitcoinService.broadcastTransaction(transaction)
  }

  // The function to transfer assets to the Litecoin network
  async transferToLitecoin (piAddress, litecoinAddress, amount) {
    // Get the PI Network balance
    const piBalance = await this.piNetworkService.getBalance(piAddress)

    // Check if there is enough balance
    if (piBalance < amount) {
      throw new Error('Insufficient balance')
    }

    // Convert the PI Network amount to Litecoin
    const litecoinAmount = this.piNetworkService.convertToLitecoin(amount)

    // Generate a Litecoin transaction
    const transaction = this.litecoinService.createTransaction(
      litecoinAddress,
      litecoinAmount
    )

    // Sign the transaction with the private key
    const privateKey = this.litecoinService.getPrivateKey(piAddress)
    this.litecoinService.signTransaction(transaction, privateKey)

    // Broadcast the transaction to the Litecoin network
    return this.litecoinService.broadcastTransaction(transaction)
  }
}

module.exports = BlockchainAdapterService
