module.exports = {
  networks: {
    development: {
      host: "localhost",
      port: 8545,
      network_id: "*",
      gas: 5000000,
      gasPrice: 20000000000
    },
    testnet: {
      host: "testnet.pi.network",
      port: 8545,
      network_id: "testnet",
      gas: 5000000,
      gasPrice: 20000000000
    },
    mainnet: {
      host: "mainnet.pi.network",
      port: 8545,
      network_id: "mainnet",
      gas: 5000000,
      gasPrice: 20000000000
    }
  },
  compilers: {
    solc: {
      version: "0.8.0",
      settings: {
        optimizer: {
          enabled: true,
          runs: 200
        }
      }
    }
  }
};
