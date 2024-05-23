const Web3 = require('web3')
const MultiSigWallet = require('../contracts/MultiSigWallet.json')

class MultiSigWalletService {
  constructor (web3, contractAddress) {
    this.web3 = web3
    this.contract = new this.web3.eth.Contract(
      MultiSigWallet.abi,
      contractAddress
    )
  }

  // The function to create a new multi-signature wallet
  createWallet (owners, requiredSignatures) {
    return new Promise((resolve, reject) => {
      this.contract.methods.createWallet(owners, requiredSignatures).send(
        {
          from: this.web3.eth.defaultAccount
        },
        (error, result) => {
          if (error) {
            reject(error)
          } else {
            resolve(result)
          }
        }
      )
    })
  }

  // The function to submit a new transaction
  submitTransaction (to, value, data) {
    return new Promise((resolve, reject) => {
      this.contract.methods.submitTransaction(to, value, data).send(
        {
          from: this.web3.eth.defaultAccount
        },
        (error, result) => {
          if (error) {
            reject(error)
          } else {
            resolve(result)
          }
        }
      )
    })
  }

  // The function to authorize a transaction
  authorizeTransaction (transactionId) {
    return new Promise((resolve, reject) => {
      this.contract.methods.authorizeTransaction(transactionId).send(
        {
          from: this.web3.eth.defaultAccount
        },
        (error, result) => {
          if (error) {
            reject(error)
          } else {
            resolve(result)
          }
        }
      )
    })
  }

  // The function to execute a transaction
  executeTransaction (transactionId) {
    return new Promise((resolve, reject) => {
      this.contract.methods.executeTransaction(transactionId).send(
        {
          from: this.web3.eth.defaultAccount
        },
        (error, result) => {
          if (error) {
            reject(error)
          } else {
            resolve(result)
          }
        }
      )
    })
  }

  // The function to get the list of pending transactions
  getPendingTransactions () {
    return new Promise((resolve, reject) => {
      this.contract.methods
        .getPendingTransactions()
        .call({}, (error, result) => {
          if (error) {
            reject(error)
          } else {
            resolve(result)
          }
        })
    })
  }
}

module.exports = MultiSigWalletService
