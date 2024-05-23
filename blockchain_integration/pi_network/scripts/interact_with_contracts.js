const { ethers } = require('hardhat')

async function interactWithContracts () {
  const piNToken = await ethers.getContract('PiNToken')
  const piNetworkBridge = await ethers.getContract('PiNetworkBridge')
  const deployer = await ethers.getSigner()

  // Example: Transfer tokens
  await piNToken.transfer('user address here', '1000000000000000000', {
    from: deployer.address
  })

  // Example: Deposit tokens
  await piNetworkBridge.deposit('100000000000000000', {
    from: deployer.address
  })

  // Example: Withdraw tokens
  await piNetworkBridge.withdraw('100000000000000000', {
    from: deployer.address
  })
}

interactWithContracts()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error)
    process.exit(1)
  })
