const ContractSphere = artifacts.require("ContractSphere");

module.exports = async function(deployer) {
  await deployer.deploy(ContractSphere);
};
