const { deployContract, getChainId } = require('@pi-nexus/contract-deployer');
const { ShipmentContract } = require('../blockchain_integration/contracts');
const { getProvider } = require('../utils/provider');

async function migrate() {
  console.log('Running shipment contract migration...');

  const provider = getProvider();
  const chainId = await getChainId(provider);
  const network = await provider.getNetwork();

  console.log(`Connected to ${network.name} (${chainId})`);

  // Deploy the ShipmentContract
  const shipmentContract = await deployContract(
    ShipmentContract,
    'ShipmentContract',
    ['createShipment', 'updateShipmentLocation'],
    {
      gas: 5000000,
      gasPrice: 20000000000,
    }
  );

  console.log(`ShipmentContract deployed at ${shipmentContract.address}`);

  // Set the contract address in the provider
  provider.setContractAddress('ShipmentContract', shipmentContract.address);

  console.log('Shipment contract migration complete!');
}

migrate().catch((error) => {
  console.error(error);
  process.exit(1);
});
