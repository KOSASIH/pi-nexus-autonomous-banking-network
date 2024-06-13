module.exports = {
  networks: {
    development: {
      host: "127.0.0.1",
      port: 8545,
      network_id: "*", // match any network
      websockets: true
    },
    live: {
      host: "178.25.19.88", // Random IP for example purposes (do not use)
      port: 80,
      network_id: 1,        // Ethereum public network
      gas: 2000000,
      gasPrice: 20e9,
      maxFeePerGas: 20e9,
      maxPriorityFeePerGas: 1e9,
      from: "0x...", // default address to use for any transaction Truffle makes during migrations
      provider: () => new Web3.providers.HttpProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"),
      production: true,
      skipDryRun: false,
      confirmations: 2,
      timeoutBlocks: 50,
      deploymentPollingInterval: 1000,
      networkCheckTimeout: 5000,
      disableConfirmationListener: false
    }
  },
  solidity: {
    version: "0.8.10",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  plugins: ["truffle-plugin-verify"],
  api_keys: {
    etherscan: "YOUR_ETHERSCAN_API_KEY"
  }
};
