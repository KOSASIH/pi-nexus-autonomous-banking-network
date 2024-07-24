const DEX = artifacts.require("DEX");

module.exports = async function(deployer) {
    await deployer.deploy(DEX);
};
