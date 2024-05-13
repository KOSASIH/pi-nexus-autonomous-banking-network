const { deployments, ethers } = require('hardhat');

module.exports = async function () {
  const { deploy } = deployments;

  await deploy('MultiChainBank', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    log: true,
    dependsOn: ['SafeMath', 'Address', 'Strings'],
  });

  await deploy('MultiChainBankManager', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    log: true,
    dependsOn: ['SafeMath', 'Address', 'Strings'],
  });

  await deploy('MultiChainToken', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    log: true,
    dependsOn: ['SafeMath', 'Address', 'Strings'],
  });

  await deploy('MultiChainOracle', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    log: true,
    dependsOn: ['SafeMath', 'Address', 'Strings'],
  });

  await deploy('MultiChainExchange', {
    from: await ethers.getNamedSigner('deployer').getAddress(),
    log: true,
    dependsOn: ['SafeMath', 'Address', 'Strings', 'MultiChainToken', 'MultiChainOracle'],
  });
};

module.exports.tags = ['MultiChainBank', 'MultiChainBankManager', 'MultiChainToken', 'MultiChainOracle', 'MultiChainExchange'];
