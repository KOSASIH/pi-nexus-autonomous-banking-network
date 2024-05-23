const { deployments, ethers } = require('hardhat')

async function deployPiNetworkBridge () {
  await deployments.fixture(['PiNetworkBridge'])
  const piNetworkBridge = await ethers.getContract('PiNetworkBridge')
  const piTokenAddress = 'PiToken address here' // Replace with the actual PiToken contract address

  await deployments.deploy('PiNetworkBridge', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    args: [piTokenAddress],
    log: true
  })
}

module.exports = deployPiNetworkBridge
module.exports.tags = ['PiNetworkBridge']
