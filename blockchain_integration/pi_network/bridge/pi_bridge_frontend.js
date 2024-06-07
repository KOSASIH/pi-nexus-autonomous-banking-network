import Web3 from "web3";
import { PiBridgeContract } from "./pi_bridge_contract";

const web3 = new Web3(window.ethereum);
const piBridgeContract = new PiBridgeContract("0x...PiBridgeContractAddress...");

async function depositPiTokens(amount) {
  try {
    const txHash = await piBridgeContract.deposit(amount, { from: "0x...UserAddress..." });
    console.log(`Deposit successful: ${txHash}`);
  } catch (error){
    console.error(`Deposit failed: ${error}`);
  }
}

async function withdrawPiTokens(amount) {
  try {
    const txHash = await piBridgeContract.withdraw(amount, { from: "0x...UserAddress..." });
    console.log(`Withdrawal successful: ${txHash}`);
  } catch (error) {
    console.error(`Withdrawal failed: ${error}`);
  }
}

async function getBalance() {
  try {
    const balance = await piBridgeContract.userBalances("0x...UserAddress...");
    console.log(`Balance: ${balance}`);
  } catch (error) {
    console.error(`Failed to retrieve balance: ${error}`);
  }
}

// Example usage:
depositPiTokens(10);
withdrawPiTokens(5);
getBalance();
