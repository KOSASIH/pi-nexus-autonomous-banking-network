import Web3 from 'web3';
import { DEXContract } from './DEXContract.json';

const web3 = new Web3(window.ethereum);

const dexContract = new web3.eth.Contract(
  DEXContract.abi,
  '0x...DEXContractAddress...'
);

// Function to set the exchange rate for a token pair
async function setExchangeRate(tokenIn, tokenOut, rate) {
  try {
    const tx = await dexContract.methods.setExchangeRate(tokenIn, tokenOut, rate).send({
      from: web3.eth.accounts[0],
    });
    console.log(`Exchange rate set: ${tx.transactionHash}`);
  } catch (error) {
    console.error(`Error setting exchange rate: ${error.message}`);
  }
}

// Function to execute a trade
async function trade(tokenIn, tokenOut, amountIn) {
  try {
    const tx = await dexContract.methods.trade(tokenIn, tokenOut, amountIn).send({
      from: web3.eth.accounts[0],
    });
    console.log(`Trade executed: ${tx.transactionHash}`);
  } catch (error) {
    console.error(`Error executing trade: ${error.message}`);
  }
}

// Function to get trade details
async function getTrade(index) {
  try {
    const trade = await dexContract.methods.getTrade(index).call();
    console.log(`Trade details: ${JSON.stringify(trade)}`);
  } catch (error) {
    console.error(`Error getting trade details: ${error.message}`);
  }
}

// Function to get the number of trades
async function getTradeCount() {
  try {
    const count = await dexContract.methods.getTradeCount().call();
    console.log(`Number of trades: ${count}`);
  } catch (error) {
    console.error(`Error getting trade count: ${error.message}`);
  }
}
