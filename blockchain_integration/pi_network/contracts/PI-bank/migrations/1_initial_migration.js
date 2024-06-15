const PIBank = artifacts.require("PIBank");

module.exports = async function(deployer) {
  await deployer.deploy(PIBank);
  const pibank = await PIBank.deployed();
  console.log(`PIBank deployed at ${pibank.address}`);
};
