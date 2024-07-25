// truffle-config.js

module.exports = {
  // Network settings
  networks: {
    development: {
      host: "localhost",
      port: 8545,
      network_id: "*", // Match any network id
      gas: 8000000,
      gasPrice: 20000000000,
    },
    ropsten: {
      provider: () => new Web3.providers.HttpProvider("https://ropsten.infura.io/v3/YOUR_PROJECT_ID"),
      network_id: 3,
      gas: 8000000,
      gasPrice: 20000000000,
    },
    mainnet: {
      provider: () => new Web3.providers.HttpProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"),
      network_id: 1,
      gas: 8000000,
      gasPrice: 20000000000,
    },
  },

  // Compiler settings
  compilers: {
    solc: {
      version: "0.8.10",
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

  // Testing settings
  testing: {
    testMatch: ["**/*.test.js"],
  },

  // Contracts settings
  contracts_build_directory: "./build/contracts",
  contracts_directory: "./contracts",

  // Plugins settings
  plugins: ["truffle-plugin-verify"],

  // API settings
  api_host: "127.0.0.1",
  api_port: 8546,
};
