// Import required libraries
const { ethers } = require("ethers");
const { deployContract } = require("ethereum-waffle");
const { readFileSync } = require("fs");
const { join } = require("path");

// Set the provider for the Pi network
const provider = new ethers.providers.JsonRpcProvider("https://rpc.pi.network");

// Set the deployer's wallet
const deployerWallet = new ethers.Wallet("0x...", provider);

// Set the gas price and gas limit for deployments
const gasPrice = ethers.utils.parseUnits("20", "gwei");
const gasLimit = 8000000;

// Load the compiled contract artifacts
const exchangeContractArtifact = readFileSync(
  join(__dirname, "../contracts/ExchangeContract.sol/ExchangeContract.json"),
);
const factoryContractArtifact = readFileSync(
  join(__dirname, "../contracts/FactoryContract.sol/FactoryContract.json"),
);

// Deploy the factory contract
async function deployFactoryContract() {
  const factoryContract = await deployContract(
    deployerWallet,
    factoryContractArtifact,
    {
      gasPrice,
      gasLimit,
    },
  );
  console.log(`Factory contract deployed to ${factoryContract.address}`);
  return factoryContract.address;
}

// Deploy the exchange contract
async function deployExchangeContract(factoryContractAddress) {
  const exchangeContract = await deployContract(
    deployerWallet,
    exchangeContractArtifact,
    {
      gasPrice,
      gasLimit,
      args: [factoryContractAddress],
    },
  );
  console.log(`Exchange contract deployed to ${exchangeContract.address}`);
  return exchangeContract.address;
}

// Main deployment function
async function deploy() {
  const factoryContractAddress = await deployFactoryContract();
  const exchangeContractAddress = await deployExchangeContract(
    factoryContractAddress,
  );

  // Set the exchange contract address in the factory contract
  const factoryContract = new ethers.Contract(
    factoryContractAddress,
    factoryContractArtifact.abi,
    deployerWallet,
  );
  await factoryContract.setExchangeContract(exchangeContractAddress);

  console.log("Deployment complete!");
}

// Run the deployment script
deploy().catch((error) => {
  console.error(error);
  process.exit(1);
});
