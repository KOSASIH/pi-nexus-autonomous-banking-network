// truffle-config.js
module.exports = {
  // Network settings
  networks: {
    development: {
      host: "127.0.0.1",
      port: 8545,
      network_id: "*",
      gas: 6721975,
      gasPrice: 20000000000,
    },
    test: {
      host: "127.0.0.1",
      port: 8545,
      network_id: "*",
      gas: 6721975,
      gasPrice: 20000000000,
    },
    mainnet: {
      provider: () => new Web3.providers.HttpProvider("https://mainnet.infura.io/v3/454c372bb595486f90fc6295b128695c"),
      network_id: 1,
      gas: 6721975,
      gasPrice: 20000000000,
    },
  },

  // Compiler settings
  compilers: {
    solc: {
      version: "0.8.0",
      settings: {
        optimizer: {
          enabled: true,
          runs: 200,
        },
      },
    },
  },

  // Migrations settings
  migrations: {
    deployer: "truffle-deployer",
  },

  // Plugins settings
  plugins: ["truffle-plugin-verify"],

  // API settings
  api: {
    host: "127.0.0.1",
    port: 8546,
  },
};
