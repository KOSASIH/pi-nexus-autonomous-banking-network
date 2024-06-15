const path = require("path");

module.exports = {
  networks: {
    development: {
      host: "127.0.0.1",
      port: 8545,
      network_id: "*", // Any network (default: none)
      gas: 6721975,
      gasPrice: 20000000000,
    },
    hardhat: {
      forking: {
        url: "http://localhost:8545",
      },
    },
  },
  mocha: {
    timeout: 100000,
  },
  compilers: {
    solc: {
      version: "0.8.4", //Change to the version you want to use
      settings: {
        optimizer: {
          enabled: true,
          runs: 200,
        },
      },
    },
  },
  db: {
    enabled: false,
  },
  plugins: ["truffle-plugin-verify"],
  contracts_build_directory: path.join(__dirname, "build/contracts"),
  contracts_directory: path.join(__dirname, "contracts"),
};
