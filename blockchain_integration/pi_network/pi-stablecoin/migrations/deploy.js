const { ethers } = require('hardhat');
const { deploy } = require('@openzeppelin/hardhat-upgrades');

async function deployContracts() {
  console.log('Deploying contracts...');

  // Deploy PiOracle contract
  const PiOracle = await ethers.getContractFactory('PiOracle');
  const piOracle = await deploy('PiOracle', {
    args: [],
    log: true,
  });
  console.log(`PiOracle contract deployed to ${piOracle.address}`);

  // Deploy PiStablecoin contract
  const PiStablecoin = await ethers.getContractFactory('PiStablecoin');
  const piStablecoin = await deploy('PiStablecoin', {
    args: [piOracle.address],
    log: true,
  });
  console.log(`PiStablecoin contract deployed to ${piStablecoin.address}`);

  // Set the PiOracle contract as the oracle for the PiStablecoin contract
  await piStablecoin.setPiOracle(piOracle.address);
  console.log('PiOracle contract set as the oracle for PiStablecoin contract');

  // Mint initial supply of Pi tokens to the deployer
  const initialSupply = ethers.utils.parseEther('1000000');
  await piStablecoin.mint(process.env.DEPLOYER_ADDRESS, initialSupply);
  console.log(`Initial supply of ${initialSupply} Pi tokens minted to ${process.env.DEPLOYER_ADDRESS}`);

  console.log('Contracts deployed successfully!');
}

deployContracts()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
