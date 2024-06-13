const DAO = artifacts.require("DAO");
const Governance = artifacts.require("Governance");
const Token = artifacts.require("Token");

module.exports = async function (deployer) {
  await deployer.deploy(DAO);
  await deployer.deploy(Governance);
  await deployer.deploy(Token);
};
