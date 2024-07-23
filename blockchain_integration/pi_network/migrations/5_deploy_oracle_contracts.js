const PiNetworkOracle = artifacts.require('PiNetworkOracle');

module.exports = async function (deployer) {
  await deployer.deploy(PiNetworkOracle);
  const piNetworkOracle = await PiNetworkOracle.deployed();
  console.log(`PiNetworkOracle deployed at ${piNetworkOracle.address}`);
};
