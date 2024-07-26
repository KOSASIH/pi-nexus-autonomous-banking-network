import Web3 from 'web3';

class Web3Utils {
  static toWei(amount, unit) {
    return Web3.utils.toWei(amount, unit);
  }

  static fromWei(amount, unit) {
    return Web3.utils.fromWei(amount, unit);
  }

  static toHex(string) {
    return Web3.utils.toHex(string);
  }

  static fromHex(hex) {
    return Web3.utils.fromHex(hex);
  }

  static soliditySha3(...args) {
    return Web3.utils.soliditySha3(...args);
  }

  static keccak256(data) {
    return Web3.utils.keccak256(data);
  }

  static asciiToHex(string) {
    return Web3.utils.asciiToHex(string);
  }

  static hexToAscii(hex) {
    return Web3.utils.hexToAscii(hex);
  }

  static isAddress(address) {
    return Web3.utils.isAddress(address);
  }

  static isValidAddress(address) {
    return Web3.utils.isValidAddress(address);
  }

  static toChecksumAddress(address) {
    return Web3.utils.toChecksumAddress(address);
  }

  static getTransactionCount(address) {
    return Web3.eth.getTransactionCount(address);
  }

  static getBlockNumber() {
    return Web3.eth.getBlockNumber();
  }

  static getBlockByNumber(blockNumber) {
    return Web3.eth.getBlockByNumber(blockNumber);
  }

  static getTransactionByHash(transactionHash) {
    return Web3.eth.getTransactionByHash(transactionHash);
  }

  static getTransactionReceipt(transactionHash) {
    return Web3.eth.getTransactionReceipt(transactionHash);
  }
}

export default Web3Utils;
