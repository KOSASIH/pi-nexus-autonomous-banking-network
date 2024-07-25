// 1_initial_migration.js

const PiAIMarketplace = artifacts.require("PiAIMarketplace");

module.exports = async function(deployer) {
  await deployer.deploy(PiAIMarketplace);

  // Initialize the marketplace with some default values
  const marketplace = await PiAIMarketplace.deployed();
  await marketplace.initialize(
    "PiAI Marketplace",
    "A decentralized marketplace for AI models, datasets, and algorithms",
    "https://pi-ai-marketplace.io"
  );

  console.log("PiAIMarketplace contract deployed and initialized!");
};
