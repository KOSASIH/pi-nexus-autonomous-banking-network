const bitcoin = require('bitcoinjs-lib')
const { ChainId, ChainName } = require('../constants')

class BitcoinAdapter {
  constructor () {
    this.network = bitcoin.networks.bitcoin
    this.chainId = ChainId.BITCOIN
    this.chainName = ChainName.BITCOIN
  }

  async getBalance (address) {
    const blockchain = new bitcoin.Blockchain.Blockchain(this.network)
    const balance = await blockchain.getBalance(address)
    return balance
  }

  async sendTransaction (fromAddress, toAddress, value) {
    const privateKey = Buffer.from(process.env.BITCOIN_PRIVATE_KEY, 'hex')
    const publicKey = bitcoin.ECPair.fromPrivateKey(privateKey).publicKey
    const address = bitcoin.payments.p2pkh({
      pubkey: publicKey,
      network: this.network
    }).address
    const transaction = new bitcoin.TransactionBuilder(this.network)
    transaction.addInput(fromAddress, 0xffffffff)
    transaction.addOutput(toAddress, value)
    transaction.sign(0, privateKey)
    const rawTransaction = transaction.build().toHex()
    const receipt = await this.broadcastTransaction(rawTransaction)
    return receipt
  }

  async getTransactionReceipt (transactionHash) {
    const blockchain = new bitcoin.Blockchain.Blockchain(this.network)
    const receipt = await blockchain.getTransaction(transactionHash)
    return receipt
  }

  async getBlockByNumber (blockNumber) {
    const blockchain = new bitcoin.Blockchain.Blockchain(this.network)
    const block = await blockchain.getBlock(blockNumber)
    return block
  }

  async broadcastTransaction (rawTransaction) {
    const client = new bitcoin.RPCPeer('http://localhost:18332')
    const receipt = await client.sendCommand('sendrawtransaction', [
      rawTransaction
    ])
    return receipt
  }
}

module.exports = BitcoinAdapter
