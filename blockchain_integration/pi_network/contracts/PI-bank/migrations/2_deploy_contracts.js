const { deployer, web3 } = require('@openzeppelin/truffle-deployer');

module.exports = async function(deployer) {
  // Initialize the deployer
  await deployer.initialize();

  // Create a new migration
  const migration = await deployer.createMigration('2_deploy_contracts');

  // Set the network ID
  const networkId = await web3.eth.net.getId();

  // Set the gas price
  const gasPrice = await web3.eth.getGasPrice();

  // Set the gas limit
  const gasLimit = 8000000;

  // Deploy the contracts
  await deployer.deploy([
    {
      contract: 'PIBankAccessControl',
      args: [],
      gas: gasLimit,
      gasPrice: gasPrice,
    },
    {
      contract: 'PIBank Gamification',
      args: [],
      gas: gasLimit,
      gasPrice: gasPrice,
    },
    {
      contract: 'PIBankRegulatoryCompliance',
      args: [],
      gas: gasLimit,
      gasPrice: gasPrice,
    },
  ]);

  // Link the contracts
  await deployer.link('PIBankAccessControl', 'PIBank');
  await deployer.link('PIBankGamification', 'PIBank');
  await deployer.link('PIBankRegulatoryCompliance', 'PIBank');

  // Save the migration
  await migration.save();
};
