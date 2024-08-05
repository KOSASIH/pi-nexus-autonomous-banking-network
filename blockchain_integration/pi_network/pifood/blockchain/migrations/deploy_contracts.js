const { deployContract, getChainId, getGasPrice, getBlockNumber } = require('@openzeppelin/truffle-deployer');
const { ethers } = require('ethers');
const { readFileSync } = require('fs');
const { join } = require('path');

const CONTRACTS_DIR = './contracts';
const BUILD_DIR = './build';
const DEPLOYMENT_DIR = './deployment';

const CONTRACT_NAMES = [
  'InventoryManager',
  'ERC721Token',
  'ERC20Token',
];

const NETWORKS = {
  mainnet: {
    provider: () => new ethers.providers.InfuraProvider('mainnet', 'YOUR_INFURA_PROJECT_ID'),
    gasPrice: ethers.utils.parseUnits('20', 'gwei'),
  },
  ropsten: {
    provider: () => new ethers.providers.InfuraProvider('ropsten', 'YOUR_INFURA_PROJECT_ID'),
    gasPrice: ethers.utils.parseUnits('10', 'gwei'),
  },
  local: {
    provider: () => new ethers.providers.JsonRpcProvider('http://localhost:8545'),
    gasPrice: ethers.utils.parseUnits('1', 'gwei'),
  },
};

async function deployContracts(network) {
  const provider = NETWORKS[network].provider();
  const gasPrice = NETWORKS[network].gasPrice;
  const chainId = await getChainId(provider);
  const blockNumber = await getBlockNumber(provider);

  console.log(`Deploying contracts to ${network} network (chainId: ${chainId}, blockNumber: ${blockNumber})`);

  for (const contractName of CONTRACT_NAMES) {
    const contractPath = join(CONTRACTS_DIR, `${contractName}.sol`);
    const contractBuildPath = join(BUILD_DIR, `${contractName}.json`);
    const contractDeploymentPath = join(DEPLOYMENT_DIR, `${contractName}.json`);

    const contractArtifact = require(contractBuildPath);
    const contractBytecode = contractArtifact.bytecode;
    const contractAbi = contractArtifact.abi;

    const deployer = await deployContract(contractName, contractBytecode, contractAbi, provider, gasPrice);

    console.log(`Deployed ${contractName} contract to ${deployer.contractAddress}`);

    const deploymentData = {
      contractAddress: deployer.contractAddress,
      abi: contractAbi,
    };

    fs.writeFileSync(contractDeploymentPath, JSON.stringify(deploymentData, null, 2));
  }
}

module.exports = {
  deployContracts,
};
