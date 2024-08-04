const PiCoin = artifacts.require("PiCoin");

module.exports = async function(deployer) {
  await deployer.deploy(PiCoin);
  const piCoin = await PiCoin.deployed();
  console.log(`Pi Coin deployed at ${piCoin.address}`);
};
