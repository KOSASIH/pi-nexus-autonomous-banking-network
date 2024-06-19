const { upgradeContract, getChainId } = require('@pi-nexus/contract-deployer');
const { LogisticsContract } = require('../blockchain_integration/contracts');
const { getProvider } = require('../utils/provider');

async function migrate() {
  console.log('Running logistics contract migration...');

  const provider = getProvider();
  const chainId = await getChainId(provider);
  const network = await provider.getNetwork();

  console.log(`Connected to ${network.name} (${chainId})`);

  // Get the existing LogisticsContract address
  const logisticsContractAddress = provider.getContractAddress('LogisticsContract');

  // Upgrade the LogisticsContract
  const upgradedLogisticsContract = await upgradeContract(
    LogisticsContract,
    logisticsContractAddress,
    ['addShipmentTracking', 'updateShipmentStatus'],
    {
      gas: 5000000,
      gasPrice: 20000000000,
    }
  );

  console.log(`LogisticsContract upgraded at ${upgradedLogisticsContract.address}`);

  // Set the upgraded contract address in the provider
  provider.setContractAddress('LogisticsContract', upgradedLogisticsContract.address);

  console.log('Logistics contract migration complete!');
}

migrate().catch((error) => {
  console.error(error);
  process.exit(1);
});
