import Web3 from 'web3';
import { RiskManagementContract } from './RiskManagementContract.json';
import { RiskManagementAIModel } from './RiskManagementAIModel.json';

const web3 = new Web3(window.ethereum);

const riskManagementContract = new web3.eth.Contract(
  RiskManagementContract.abi,
  '0x...RiskManagementContractAddress...'
);

const riskManagementAIModel = new web3.eth.Contract(
  RiskManagementAIModel.abi,
  '0x...RiskManagementAIModelAddress...'
);

// Function to update a user's risk score
async function updateRiskScore(userAddress, riskScore) {
  try {
    const tx = await riskManagementContract.methods.updateRiskScore(userAddress, riskScore).send({
      from: userAddress,
    });
    console.log(`Risk score updated: ${tx.transactionHash}`);
  } catch (error) {
    console.error(`Error updating risk score: ${error.message}`);
  }
}

// Function to update a user's portfolio
async function updatePortfolio(userAddress, portfolio) {
  try {
    const tx = await riskManagementContract.methods.updatePortfolio(userAddress, portfolio).send({
      from: userAddress,
    });
    console.log(`Portfolio updated: ${tx.transactionHash}`);
  } catch (error) {
    console.error(`Error updating portfolio: ${error.message}`);
  }
}

// Function to calculate a user's risk score based on their portfolio
async function calculateRiskScore(userAddress) {
  try {
    const riskScore = await riskManagementAIModel.methods.calculateRiskScore(userAddress).call();
    console.log(`Risk score: ${riskScore}`);
  } catch (error) {
    console.error(`Error calculating risk score: ${error.message}`);
  }
}
