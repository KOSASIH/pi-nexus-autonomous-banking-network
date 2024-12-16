// scripts/interact.js
const { ethers } = require("hardhat");

async function main() {
    const multiSigWalletAddress = "0xYourMultiSigWalletAddress"; // Replace with actual address
    const GovernanceContract = await ethers.getContractAt("GovernanceContract", multiSigWalletAddress);

    const MultiSigWallet = await ethers.getContractAt("MultiSigWallet", multiSigWalletAddress);

    // Example: Submit a transaction to the MultiSigWallet
    const tx = await MultiSigWallet.submitTransaction("0xRecipientAddress", ethers.utils.parseEther("1.0"), "0x");
    await tx.wait();
    console.log("Transaction submitted!");

    // Example: Confirm a transaction
    const txIndex = 0; // Replace with the actual transaction index
    const confirmTx = await MultiSigWallet.confirmTransaction(txIndex);
    await confirmTx.wait();
    console.log("Transaction confirmed!");

    // Example: Pause the contract
    const pauseTx = await MultiSigWallet.pause();
    await pauseTx.wait();
    console.log("Contract paused!");

    // Example: Unpause the contract
    const unpauseTx = await MultiSigWallet.unpause();
    await unpauseTx.wait();
    console.log("Contract unpaused!");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
