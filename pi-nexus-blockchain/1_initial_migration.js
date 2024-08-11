const { deployer, web3 } = require('@openzeppelin/truffle-deployer');
const { BN } = web3.utils;

const IdentityVerification = artifacts.require('IdentityVerification');
const UserRegistry = artifacts.require('UserRegistry');

module.exports = async (deployer) => {
  // Set the gas price and gas limit for the deployment
  const gasPrice = new BN('20000000000');
  const gasLimit = new BN('5000000');

  // Deploy the IdentityVerification contract
  await deployer.deploy(IdentityVerification, {
    from: deployer.accounts[0],
    gas: gasLimit,
    gasPrice: gasPrice,
  });

  // Deploy the UserRegistry contract
  await deployer.deploy(UserRegistry, {
    from: deployer.accounts[0],
    gas: gasLimit,
    gasPrice: gasPrice,
  });

  // Set the owner of the contracts to the deployer account
  await IdentityVerification.deployed().then(instance => instance.transferOwnership(deployer.accounts[0]));
  await UserRegistry.deployed().then(instance => instance.transferOwnership(deployer.accounts[0]));
};
