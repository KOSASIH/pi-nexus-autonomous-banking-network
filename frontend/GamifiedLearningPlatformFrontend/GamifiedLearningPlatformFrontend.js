import Web3 from 'web3';
import { BlockchainBasicsContract } from './BlockchainBasicsContract.json';

const web3 = new Web3(window.ethereum);

const blockchainBasicsContract = new web3.eth.Contract(
  BlockchainBasicsContract.abi,
  '0x...BlockchainBasicsContractAddress...'
);

// Function to complete a lesson
async function completeLesson(lessonId) {
  try {
    const tx = await blockchainBasicsContract.methods.completeLesson(lessonId).send({
      from: web3.eth.accounts[0],
    });
    console.log(`Lesson completed: ${tx.transactionHash}`);
  } catch (error) {
    console.error(`Error completing lesson: ${error.message}`);
  }
}

// Function to get a user's progress
async function getProgress() {
  try {
    const progress = await blockchainBasicsContract.methods.getProgress(web3.eth.accounts[0]).call();
    console.log(`Progress: ${progress}`);
  } catch (error) {
    console.error(`Error getting progress: ${error.message}`);
  }
}
