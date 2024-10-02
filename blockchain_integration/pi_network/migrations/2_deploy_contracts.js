const PiNetworkFactory = artifacts.require('PiNetworkFactory');

module.exports = async function (deployer) {
  await deployer.deploy(PiNetworkFactory);
  const piNetworkFactory = await PiNetworkFactory.deployed();
  const piNetworkAddress = await piNetworkFactory.createPiNetwork();
  console.log(`PiNetwork deployed at ${piNetworkAddress}`);
};
