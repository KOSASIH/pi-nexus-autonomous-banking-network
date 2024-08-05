const { ethers } = require("hardhat");
const { deployContract, getContractAt } = require("@nomiclabs/hardhat-ethers");
const { parseEther } = require("ethers/lib/utils");
const { MerkleTree } = require("merkletreejs");
const keccak256 = require("keccak256");

// Import contract artifacts
const PropertyTransfer = require("../artifacts/PropertyTransfer.sol/PropertyTransfer.json");
const Token = require("../artifacts/Token.sol/Token.json");

// Set up the deployment configuration
const deploymentConfig = {
  network: "mainnet",
  gasPrice: parseEther("20.0"),
  gasLimit: 8000000,
};

// Set up the contract deployment parameters
const propertyTransferParams = {
  name: "PropertyTransfer",
  symbol: "PT",
  owner: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
};

const tokenParams = {
  name: "RealEstateToken",
  symbol: "RET",
  decimals: 18,
  totalSupply: parseEther("100000000"),
  owner: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
};

// Deploy the PropertyTransfer contract
async function deployPropertyTransfer() {
  const propertyTransferContract = await deployContract("PropertyTransfer", propertyTransferParams);
  console.log(`PropertyTransfer contract deployed to ${propertyTransferContract.address}`);
  return propertyTransferContract;
}

// Deploy the Token contract
async function deployToken() {
  const tokenContract = await deployContract("Token", tokenParams);
  console.log(`Token contract deployed to ${tokenContract.address}`);
  return tokenContract;
}

// Create a Merkle tree for the property metadata
async function createMerkleTree() {
  const propertyMetadata = [
    {
      id: 1,
      name: "Property 1",
      description: "This is a beautiful property",
      location: "New York",
      price: parseEther("1000.0"),
    },
    {
      id: 2,
      name: "Property 2",
      description: "This is a luxurious property",
      location: "Los Angeles",
      price: parseEther("2000.0"),
    },
    //...
  ];

  const leaves = propertyMetadata.map((property) => keccak256(property));
  const tree = new MerkleTree(leaves, keccak256, { sort: true });
  const root = tree.getRoot();
  console.log(`Merkle tree root: ${root}`);
  return root;
}

// Deploy the contracts and create the Merkle tree
async function main() {
  const propertyTransferContract = await deployPropertyTransfer();
  const tokenContract = await deployToken();
  const merkleTreeRoot = await createMerkleTree();

  // Set the Merkle tree root on the PropertyTransfer contract
  await propertyTransferContract.setMerkleTreeRoot(merkleTreeRoot);

  // Set the Token contract address on the PropertyTransfer contract
  await propertyTransferContract.setTokenContractAddress(tokenContract.address);

  console.log("Contracts deployed and configured successfully!");
}

main()
 .then(() => process.exit(0))
 .catch((error) => {
    console.error(error);
    process.exit(1);
  });
