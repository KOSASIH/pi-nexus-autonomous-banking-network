const { ethers } = require('hardhat');

async function main() {
  const [deployer] = await ethers.getSigners();

  console.log('Deploying contracts with the account:', deployer.address);

  const SafeMath = await ethers.getContractFactory('SafeMath');
  const address = await SafeMath.deploy();
  await address.deployed();
  console.log('SafeMath deployed to:', address.address);

  const Strings = await ethers.getContractFactory('Strings');
  const strings = await Strings.deploy();
  await strings.deployed();
  console.log('Strings deployed to:', strings.address);

  const MultiChainBank = await ethers.getContractFactory('MultiChainBank');
  const multiChainBank = await MultiChainBank.deploy();
  await multiChainBank.deployed();
  console.log('MultiChainBank deployed to:', multiChainBank.address);

  const MultiChainBankManager = await ethers.getContractFactory('MultiChainBankManager');
  const multiChainBankManager = await MultiChainBankManager.deploy();
  await multiChainBankManager.deployed();
  console.log('MultiChainBankManager deployed to:', multiChainBankManager.address);

  const MultiChainToken = await ethers.getContractFactory('MultiChainToken');
  const multiChainToken = await MultiChainToken.deploy();
  await multiChainToken.deployed();
  console.log('MultiChainToken deployed to:', multiChainToken.address);

  const MultiChainOracle = await ethers.getContractFactory('MultiChainOracle');
  const multiChainOracle = await MultiChainOracle.deploy();
  await multiChainOracle.deployed();
  console.log('MultiChainOracle deployed to:', multiChainOracle.address);

  const MultiChainExchange = await ethers.getContractFactory('MultiChainExchange');
  const multiChainExchange = await MultiChainExchange.deploy(multiChainToken.address, multiChainOracle.address);
  await multiChainExchange.deployed();
  console.log('MultiChainExchange deployed to:', multiChainExchange.address);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
