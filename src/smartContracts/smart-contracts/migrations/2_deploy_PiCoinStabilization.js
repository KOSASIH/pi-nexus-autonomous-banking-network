const PiCoinStabilization = artifacts.require("PiCoinStabilization");

module.exports = function (deployer) {
    // Deploy the PiCoinStabilization contract
    deployer.deploy(PiCoinStabilization);
};
