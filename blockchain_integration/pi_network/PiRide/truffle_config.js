module.exports = {
  // Network settings
  networks: {
    development: {
      host: "localhost",
      port: 8545,
      network_id: "*", // Match any network id
      gas: 8000000,
      gasPrice: 20000000000
    },
    test: {
      host: "localhost",
      port: 8545,
      network_id: "*", // Match any network id
      gas: 8000000,
      gasPrice: 20000000000
    }
  },

  // Compiler settings
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
  },

  // Migrations settings
  migrations_directory: "./migrations",
  migrations: ["1_initial_migration.js", "2_deploy_contracts.js"],

  // Testing settings
  test_file_extension_regexp: /.*\.js$/,
  test_directory: "./test",

  // Web3 settings
  web3: {
    provider: () => new Web3.providers.HttpProvider("http://localhost:8545"),
    network_id: 5777
  }
};
