// 2_add_reputation_contract.js

const PiAIMarketplace = artifacts.require("PiAIMarketplace");
const Reputation = artifacts.require("Reputation");

module.exports = async function(deployer) {
  // Deploy the Reputation contract
  await deployer.deploy(Reputation);

  // Set the Reputation contract address in the PiAIMarketplace contract
  const marketplace = await PiAIMarketplace.deployed();
  const reputation = await Reputation.deployed();
  await marketplace.setReputationContractAddress(reputation.address);

  console.log("Reputation contract deployed and set in PiAIMarketplace!");
};
