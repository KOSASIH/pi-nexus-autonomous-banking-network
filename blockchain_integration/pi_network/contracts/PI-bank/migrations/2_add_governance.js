const PIBank = artifacts.require("PIBank");
const PIBankGovernance = artifacts.require("PIBankGovernance");

module.exports = async function(deployer) {
  await deployer.deploy(PIBankGovernance);
  const pibankGovernance = await PIBankGovernance.deployed();
  const pibank = await PIBank.deployed();
  await pibank.setGovernanceAddress(pibankGovernance.address);
  console.log(`PIBankGovernance deployed at ${pibankGovernance.address}`);
};
