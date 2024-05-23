const { deployments, ethers } = require('hardhat')

async function deployContracts () {
  await deployments.fixture(['PiNToken', 'PiNetworkBridge'])
  const piNToken = await ethers.getContract('PiNToken')
  const piNetworkBridge = await ethers.getContract('PiNetworkBridge')

  console.log('PiNToken address:', piNToken.address)
  console.log('PiNetworkBridge address:', piNetworkBridge.address)
}

deployContracts()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })
