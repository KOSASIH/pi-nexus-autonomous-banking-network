const PiNetworkData = artifacts.require('PiNetworkData');

module.exports = async function (deployer) {
  await deployer.deploy(PiNetworkData);
  const piNetworkData = await PiNetworkData.deployed();
  console.log(`PiNetworkData deployed at ${piNetworkData.address}`);
};
