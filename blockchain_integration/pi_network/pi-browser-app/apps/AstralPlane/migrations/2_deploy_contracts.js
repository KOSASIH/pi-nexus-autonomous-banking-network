// 2_deploy_contracts.js
const { deploy } = require('@astralplane/deploy');
const AstralPlaneAsset = artifacts.require('AstralPlaneAsset');
const AstralPlaneMarketplace = artifacts.require('AstralPlaneMarketplace');
const VRController = artifacts.require('VRController');
const VRRenderer = artifacts.require('VRRenderer');

module.exports = async (deployer) => {
  // Set the AstralPlaneAsset contract as the asset contract for the marketplace
  await AstralPlaneMarketplace.deployed().then((marketplace) => {
    marketplace.setAssetContract(AstralPlaneAsset.address);
  });

  // Set the VRController contract as the controller contract for the renderer
  await VRRenderer.deployed().then((renderer) => {
    renderer.setControllerContract(VRController.address);
  });

  // Set the VRRenderer contract as the renderer contract for the controller
  await VRController.deployed().then((controller) => {
    controller.setRendererContract(VRRenderer.address);
  });
};
