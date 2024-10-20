import Web3 from 'web3';
import { ImpactInvestingContract } from './ImpactInvestingContract.json';

const web3 = new Web3(window.ethereum);

const impactInvestingContract = new web3.eth.Contract(
  ImpactInvestingContract.abi,
  '0x...ImpactInvestingContractAddress...'
);

// Function to add a new project
async function addProject(name, description, goal) {
  try {
    const tx = await impactInvestingContract.methods.addProject(name, description, goal).send({
      from: web3.eth.accounts[0],
    });
    console.log(`Project added: ${tx.transactionHash}`);
  } catch (error) {
    console.error(`Error adding project: ${error.message}`);
  }
}

// Function to invest in a project
async function invest(projectId, amount) {
  try {
    const tx = await impactInvestingContract.methods.invest(projectId, amount).send({
      from: web3.eth.accounts[0],
    });
    console.log(`Investment made: ${tx.transactionHash}`);
  } catch (error) {
    console.error(`Error making investment: ${error.message}`);
  }
}

// Function to get a project's details
async function getProject(projectId) {
  try {
    const project = await impactInvestingContract.methods.getProject(projectId).call();
    console.log(`Project details: ${JSON.stringify(project)}`);
  } catch (error) {
    console.error(`Error getting project details: ${error.message}`);
  }
}

// Function to get a user's investments
async function getInvestments() {
  try {
    const investments = await impactInvestingContract.methods.getInvestments(web3.eth.accounts[0]).call();
    console.log(`User  investments: ${JSON.stringify(investments)}`);
  } catch (error) {
    console.error(`Error getting user investments: ${error.message}`);
  }
}
