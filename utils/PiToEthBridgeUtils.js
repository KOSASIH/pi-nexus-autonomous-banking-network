const Web3 = require('web3')

class PiToEthBridgeUtils {
  static getPiToEthereumRate (web3, contractAddress) {
    const contract = new web3.eth.Contract(PiToEthBridge.abi, contractAddress)
    return contract.methods.conversionRate().call()
  }

  static getPiTokenAddress (web3, contractAddress) {
    const contract = new web3.eth.Contract(PiToEthBridge.abi, contractAddress)
    return contract.methods.piNetworkContract().call()
  }

  static getEthereumTokenAddress (web3, contractAddress) {
    const contract = new web3.eth.Contract(PiToEthBridge.abi, contractAddress)
    return contract.methods.ethereumContract().call()
  }
}

module.exports = PiToEthBridgeUtils
