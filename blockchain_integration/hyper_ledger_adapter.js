const { FileSystemWallet, Gateway } = require('fabric-network')
const fs = require('fs')
const path = require('path')

class HyperledgerAdapter {
  constructor (networkConfigPath, walletPath) {
    this.networkConfigPath = networkConfigPath
    this.walletPath = walletPath
  }

  async initialize () {
    this.networkConfig = JSON.parse(
      fs.readFileSync(this.networkConfigPath, 'utf8')
    )
    this.wallet = new FileSystemWallet(this.walletPath)
  }

  async connect (userName) {
    const userExists = await this.wallet.exists(userName)
    if (!userExists) {
      throw new Error(`User "${userName}" not found in the wallet.`)
    }

    const connectionProfile = this.networkConfig.connectionProfile
    const connectionOptions = this.networkConfig.connectionOptions
    const gateway = new Gateway()
    await gateway.connect(connectionProfile, connectionOptions)
    const network = await gateway.getNetwork(this.networkConfig.channelName)
    const contract = network.getContract(this.networkConfig.chaincodeName)

    return { gateway, contract }
  }

  async submitTransaction (userName, contractName, ...args) {
    const { contract } = await this.connect(userName)
    const transaction = contract.createTransaction(contractName)
    const result = await transaction.submit(...args)
    return result.toString()
  }

  async evaluateTransaction (userName, contractName, ...args) {
    const { contract } = await this.connect(userName)
    const result = await contract.evaluateTransaction(contractName, ...args)
    return result.toString()
  }
}

module.exports = HyperledgerAdapter
