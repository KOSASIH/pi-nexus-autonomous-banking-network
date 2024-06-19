const { deployContract, getChainId } = require('@pi-nexus/contract-deployer');
const { LogisticsContract } = require('../blockchain_integration/contracts');
const { getProvider } = require('../utils/provider');

async function migrate() {
  console.log('Running initial migration...');

  const provider = getProvider();
  const chainId = await getChainId(provider);
  const network = await provider.getNetwork();

  console.log(`Connected to ${network.name} (${chainId})`);

  // Deploy the LogisticsContract
  const logisticsContract = await deployContract(
    LogisticsContract,
    'LogisticsContract',
    ['LogisticsContract'],
    {
      gas: 5000000,
      gasPrice: 20000000000,
    }
  );

  console.log(`LogisticsContract deployed at ${logisticsContract.address}`);

  // Set the contract address in the provider
  provider.setContractAddress('LogisticsContract', logisticsContract.address);

  console.log('Initial migration complete!');
}

migrate().catch((error) => {
  console.error(error);
  process.exit(1);
});
