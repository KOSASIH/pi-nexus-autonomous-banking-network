const Web3 = require('web3')
const { ChainId, Token } = require('pi-bank-sdk')

class PiEthereumAdapter {
  constructor (ethereumUrl, piPrivateKey) {
    this.ethereum = new Web3(ethereumUrl)
    this.pi = new Token({
      chainId: ChainId.PI_TESTNET,
      privateKey: piPrivateKey
    })
  }

  async getPiBalance (piAddress) {
    const ethBalance = await this.ethereum.eth.getBalance(piAddress)
    const piBalance = await this.pi.getBalance(piAddress)
    return { ethBalance, piBalance }
  }

  async sendPi (fromPiAddress, toPiAddress, amount) {
    const txData = await this.pi.createSendTransaction(
      fromPiAddress,
      toPiAddress,
      amount
    )
    const signedTx = await this.pi.signTransaction(txData)
    const txHash = await this.pi.sendSignedTransaction(signedTx)
    return txHash
  }

  async sendEth (fromEthAddress, toEthAddress, amount) {
    const txData = {
      from: fromEthAddress,
      to: toEthAddress,
      value: amount
    }
    const signedTx = await this.ethereum.eth.accounts.signTransaction(
      txData,
      fromEthAddress
    )
    const txHash = await this.ethereum.eth.sendSignedTransaction(
      signedTx.rawTransaction
    )
    return txHash
  }
}

module.exports = PiEthereumAdapter
