const PiNetworkFactory = artifacts.require("PiNetworkFactory");

module.exports = async function(deployer) {
  await deployer.deploy(PiNetworkFactory);
  const piNetworkFactory = await PiNetworkFactory.deployed();
  console.log(`PiNetworkFactory deployed at ${piNetworkFactory.address}`);

  const piNetworkRouter = await piNetworkFactory.createPiNetworkRouter();
  console.log(`PiNetworkRouter deployed at ${piNetworkRouter.address}`);
};
