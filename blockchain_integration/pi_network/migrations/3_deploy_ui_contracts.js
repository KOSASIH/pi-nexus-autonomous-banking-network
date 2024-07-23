const PiNetworkUI = artifacts.require("PiNetworkUI");

module.exports = async function(deployer) {
  await deployer.deploy(PiNetworkUI);
  const piNetworkUI = await PiNetworkUI.deployed();
  console.log(`PiNetworkUI deployed at ${piNetworkUI.address}`);
};
