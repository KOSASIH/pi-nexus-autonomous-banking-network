// truffle-config.js
module.exports = {
  networks: {
    testnet: {
      provider: () => new Web3.providers.HttpProvider('https://ropsten.infura.io/v3/YOUR_PROJECT_ID'),
      network_id: 3, // Ropsten testnet
      gas: 4500000,
      gasPrice: 20e9,
    },
  },
  compilers: {
    solc: {
      version: '0.8.10',
    },
  },
};
