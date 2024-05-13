const { deployments, ethers } = require('hardhat');

module.exports = async function () {
  const { deploy } = deployments;

  await deploy('SafeMath', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    log: true,
  });

  await deploy('Address', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    log: true,
  });

  await deploy('Strings', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    log: true,
  });
};

module.exports.tags = ['SafeMath', 'Address', 'Strings'];
