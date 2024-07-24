const TradingEngine = artifacts.require("TradingEngine");

module.exports = async function(deployer) {
    await deployer.deploy(TradingEngine);
};
