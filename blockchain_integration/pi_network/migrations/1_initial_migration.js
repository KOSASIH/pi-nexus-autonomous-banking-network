const { deployer, web3 } = require('@truffle/hdwallet-provider');

module.exports = async function(deployer) {
  // Initialize the Pi Network contract
  await deployer.deploy(PiNetwork, {
    from: web3.eth.accounts[0],
    gas: 5000000,
    gasPrice: web3.utils.toWei('20', 'gwei')
  });

  // Initialize the Pi Token contract
  await deployer.deploy(PiToken, {
    from: web3.eth.accounts[0],
    gas: 5000000,
    gasPrice: web3.utils.toWei('20', 'gwei')
  });
};
