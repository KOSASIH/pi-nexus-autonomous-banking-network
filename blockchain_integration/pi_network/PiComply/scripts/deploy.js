const { ethers } = require("hardhat");
const { ChainlinkOracle } = require("../contracts/oracles/ChainlinkOracle.sol");
const { RegulatoryKnowledgeGraph } = require("../contracts/RegulatoryKnowledgeGraph.sol");

async function deploy() {
  // Set up the provider and signer
  const provider = new ethers.providers.JsonRpcProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID");
  const signer = new ethers.Wallet("0xYOUR_PRIVATE_KEY", provider);

  // Deploy the Chainlink Oracle contract
  const chainlinkOracleFactory = new ethers.ContractFactory(ChainlinkOracle.abi, ChainlinkOracle.bytecode, signer);
  const chainlinkOracle = await chainlinkOracleFactory.deploy("0xLINK_TOKEN_ADDRESS", "0xORACLE_ADDRESS");
  await chainlinkOracle.deployed();
  console.log(`Chainlink Oracle deployed to ${chainlinkOracle.address}`);

  // Deploy the Regulatory Knowledge Graph contract
  const regulatoryKnowledgeGraphFactory = new ethers.ContractFactory(RegulatoryKnowledgeGraph.abi, RegulatoryKnowledgeGraph.bytecode, signer);
  const regulatoryKnowledgeGraph = await regulatoryKnowledgeGraphFactory.deploy();
  await regulatoryKnowledgeGraph.deployed();
  console.log(`Regulatory Knowledge Graph deployed to ${regulatoryKnowledgeGraph.address}`);

  // Set up the Chainlink Oracle as the data provider for the Regulatory Knowledge Graph
  await regulatoryKnowledgeGraph.setChainlinkOracle(chainlinkOracle.address);
  console.log(`Chainlink Oracle set as data provider for Regulatory Knowledge Graph`);

  // Verify the contracts on Etherscan
  await hre.run("verify:verify", {
    address: chainlinkOracle.address,
    constructorArguments: ["0xLINK_TOKEN_ADDRESS", "0xORACLE_ADDRESS"],
  });
  await hre.run("verify:verify", {
    address: regulatoryKnowledgeGraph.address,
  });
}

deploy()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
