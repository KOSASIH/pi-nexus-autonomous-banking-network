const { deployer, web3 } = require('@openzeppelin/truffle-deployer');
const { BN } = web3.utils;

const IdentityVerification = artifacts.require('IdentityVerification');
const UserRegistry = artifacts.require('UserRegistry');

module.exports = async (deployer) => {
  // Set the gas price and gas limit for the deployment
  const gasPrice = new BN('20000000000');
  const gasLimit = new BN('5000000');

  // Get the deployed instances of the contracts
  const identityVerificationInstance = await IdentityVerification.deployed();
  const userRegistryInstance = await UserRegistry.deployed();

  // Set the address of the IdentityVerification contract in the UserRegistry contract
  await userRegistryInstance.setIdentityVerificationAddress(identityVerificationInstance.address, {
    from: deployer.accounts[0],
    gas: gasLimit,
    gasPrice: gasPrice,
  });

  // Set the address of the UserRegistry contract in the IdentityVerification contract
  await identityVerificationInstance.setUserRegistryAddress(userRegistryInstance.address, {
    from: deployer.accounts[0],
    gas: gasLimit,
    gasPrice: gasPrice,
  });
};
