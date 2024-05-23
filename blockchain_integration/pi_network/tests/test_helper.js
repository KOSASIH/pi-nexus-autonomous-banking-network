const HDWalletProvider = require('@truffle/hdwallet-provider')
require('dotenv').config()

module.exports = {
  networks: {
    development: {
      provider: () =>
        new HDWalletProvider(process.env.MNEMONIC, 'http://127.0.0.1:8545/'),
      network_id: 1337
    }
  },
  compilers: {
    solc: {
      version: '0.8.0',
      settings: {
        optimizer: {
          enabled: true,
          runs: 200
        }
      }
    }
  }
}
