const { ethers } = require("hardhat");

async function monitorBridge() {
  const piNetworkBridge = await ethers.getContract("PiNetworkBridge");
  const deployer = await ethers.getSigner();

  // Example: Check the balance of tokens in the bridge
  const bridgeBalance = await piNetworkBridge.balanceOf(piNetworkBridge.address);
  console.log("PiNToken balance in PiNetworkBridge:", bridgeBalance.toString());

  // Example: Listen for deposit events
  piNetworkBridge.on("Deposit", (from, amount, event) => {
    console.log("Deposit event:", event);
    console.log("From:", from);
    console.log("Amount:", amount.toString());
  });

  // Example: Listen for withdraw events
  piNetworkBridge.on("Withdraw", (to, amount, event) => {
    console.log("Withdraw event:", event);
    console.log("To:", to);
    console.log("Amount:", amount.toString());
  });
}

monitorBridge()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
