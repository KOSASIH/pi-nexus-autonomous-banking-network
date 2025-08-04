const { deployments, ethers } = require('hardhat')

async function deploy () {
  const { deploy } = deployments
  const accounts = await ethers.getNamedSigners()

  await deploy('PiNToken', {
    from: accounts[0].address,
    log: true,
    args: [ethers.utils.parseEther('1000000')] // 1,000,000 PiN tokens
  })
}

module.exports = deploy
module.exports.tags = ['PiNToken']
