const PiToken = artifacts.require("PiToken");
const PaymentGateway = artifacts.require("PaymentGateway");

module.exports = async (deployer) => {
  // Deploy the PiToken contract
  console.log("Deploying PiToken contract...");
  await deployer.deploy(PiToken);

  // Deploy the PaymentGateway contract
  console.log("Deploying PaymentGateway contract...");
  await deployer.deploy(PaymentGateway, PiToken.address);

  // Set the PiToken contract address in the PaymentGateway contract
  console.log("Setting PiToken contract address in PaymentGateway contract...");
  const paymentGateway = await PaymentGateway.deployed();
  await paymentGateway.setPiTokenAddress(PiToken.address);
};
