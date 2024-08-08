module.exports = {
  networks: {
    development: {
      host: "localhost",
      port: 8545,
      network_id: "*" // Match any network id
    },
    ethereum: {
      provider: () => new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'),
      network_id: 1,
      gas: 2000000,
      gasPrice: 20e9
    },
    binanceSmartChain: {
      provider: () => new Web3.providers.HttpProvider('https://bsc-dataseed.binance.org/api/v1/bc/BSC/main'),
      network_id: 56,
      gas: 2000000,
      gasPrice: 20e9
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
