// 1_initial_migration.js
const { deploy } = require('@astralplane/deploy');

module.exports = async (deployer) => {
  // Deploy the AstralPlaneAsset contract
  await deployer.deploy(AstralPlaneAsset, {
    args: ['AstralPlane Asset', 'APA'],
  });

  // Deploy the AstralPlaneMarketplace contract
  await deployer.deploy(AstralPlaneMarketplace, {
    args: [AstralPlaneAsset.address],
  });

  // Deploy the VRController contract
  await deployer.deploy(VRController, {
    args: [],
  });

  // Deploy the VRRenderer contract
  await deployer.deploy(VRRenderer, {
    args: [],
  });
};
