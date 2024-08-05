const { ethers } = require("hardhat");
const { TradeFinanceContract } = require("../contracts/TradeFinanceContract.sol");

async function migrate() {
  // Set up the provider and signer
  const provider = new ethers.providers.JsonRpcProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID");
  const signer = new ethers.Wallet("0xYOUR_PRIVATE_KEY", provider);

  // Get the TradeFinance contract
  const tradeFinanceAddress = "0xTRADEFINANCE_CONTRACT_ADDRESS";
  const tradeFinance = new ethers.Contract(tradeFinanceAddress, TradeFinance.abi, signer);

  // Migrate existing trade finance data to the new contract
  const existingContracts = await tradeFinance.getExistingContracts();
  for (const contractAddress of existingContracts) {
    const contract = new ethers.Contract(contractAddress, TradeFinanceContract.abi, signer);
    const data = await contract.getData();
    await tradeFinance.updateData(contractAddress, data);
    console.log(`Migrated data for contract ${contractAddress}`);
  }

  console.log("Migration complete");
}

migrate();
