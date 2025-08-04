"use strict";

const { FileSystemWallet, Gateway } = require("fabric-network");
const path = require("path");
const fs = require("fs");

const ccpPath = path.resolve(
  __dirname,
  "..",
  "..",
  "first-network",
  "connection-org1.json",
);

async function main() {
  try {
    // Create a new file system based wallet for managing identities.
    const walletPath = path.join(process.cwd(), "wallet");
    const wallet = new FileSystemWallet(walletPath);
    console.log(`Wallet path: ${walletPath}`);

    // Check to see if we've already enrolled the user.
    const userExists = await wallet.exists("user1");
    if (!userExists) {
      console.log(
        'An identity for the user "user1" does not exist in the wallet',
      );
      console.log("Run the registerUser.js application before retrying");
      return;
    }

    // Check to see if we've loaded the network configuration.
    const connectionProfilePath = path.resolve(ccpPath);
    const connectionProfileJSON = fs.readFileSync(
      connectionProfilePath,
      "utf8",
    );
    const connectionProfile = JSON.parse(connectionProfileJSON);

    // Load the network configuration
    const channelName = "mychannel";
    const chaincodeName = "fabcar";
    const peerNames = ["peer0.org1.example.com"];

    // Create a new gateway for connecting to our peer node.
    const gateway = new Gateway();
    await gateway.connect(connectionProfile, {
      wallet,
      identity: "user1",
      discovery: { enabled: true, asLocalhost: true },
    });

    // Get the network and contract.
    const network = await gateway.getNetwork(channelName);
    const contract = network.getContract(chaincodeName);

    // Submit transaction to create car
    const carNumber = "CAR12";
    const carColour = "black";
    const carMake = "Honda";
    const carModel = "Accord";
    const carOwner = "Tom";
    await contract.submitTransaction(
      "CreateCar",
      carNumber,
      carColour,
      carMake,
      carModel,
      carOwner,
    );
    console.log(
      `Created car ${carNumber} with colour ${carColour}, make ${carMake} and model ${carModel} for owner ${carOwner}`,
    );

    // Disconnect from the gateway.
    await gateway.disconnect();
  } catch (error) {
    console.error(`Failed to submit transaction: ${error}`);
    process.exit(1);
  }
}

main();
