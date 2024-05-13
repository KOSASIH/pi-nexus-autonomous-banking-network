const { deployments, ethers } = require("hardhat");

const { deploy } = deployments;

async function deployContracts() {
  await deployments.fixture(["contracts"]);

  const [deployer] = await ethers.getSigners();

  console.log("Deploying contracts with the account:", deployer.address);

  console.log("Account balance:", (await deployer.getBalance()).toString());

  const MyContract = await ethers.getContractFactory("MyContract");

  console.log("Deploying MyContract...");

  const myContract = await MyContract.deploy();

  console.log("Deployed MyContract at:", myContract.address);

  await myContract.deployed();

  console.log("MyContract deployed.");

  const myContractArtifact = await deployments.artifacts.get("MyContract");

  console.log("MyContract ABI:", myContractArtifact.abi);

  console.log("MyContract bytecode:", myContractArtifact.bytecode);
}

deployContracts()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
