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
    rinkeby: {
      provider: function() {
        return new Web3.providers.HttpProvider("https://rinkeby.infura.io/v3/YOUR_PROJECT_ID");
      },
      network_id: 4,
      gas: 8000000,
      gasPrice: 20000000000
    }
  },

  // Contract compilation settings
  compilers: {
    solc: {
      version: "0.8.10",
      settings: {
        optimizer: {
          enabled: true,
          runs: 200
        }
      }
    }
  },

  // Migration settings
  migrations_directory: "./migrations",
  migrations: ["1_initial_migration.js", "2_create_user_contract.js", "3_create_ride_contract.js"],

  // Test settings
  test_directory: "./test",
  test_file_extension_regexp: /.*\.js$/,
  test_compile_enabled: true
};
