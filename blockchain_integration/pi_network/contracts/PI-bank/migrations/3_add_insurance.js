const PIBank = artifacts.require("PIBank");
const PIBankInsurance = artifacts.require("PIBankInsurance");

module.exports =async function(deployer) {
  await deployer.deploy(PIBankInsurance);
  const pibankInsurance = await PIBankInsurance.deployed();
  const pibank = await PIBank.deployed();
  await pibank.setInsuranceAddress(pibankInsurance.address);
  console.log(`PIBankInsurance deployed at ${pibankInsurance.address}`);
};
