// blockchain_integration/pi_network.js
const PiNetwork = require('pi-network');

const piNetworkConfig = {
  nodeUrl: 'https://api.minepi.com/v1/node',
  walletUrl: 'https://api.minepi.com/v1/wallet',
};

const piNetwork = new PiNetwork(piNetworkConfig);

module.exports = piNetwork;
