const { ethers } = require("hardhat");

const { Contract, Wallet } = ethers;

const MyContractAddress = "0xMyContractAddress";
const MyContractABI = [
  // ABI for MyContract
];

const web3 = new ethers.providers.Web3Provider(new ethers.providers.JsonRpcProvider("http://localhost:8545"));

const myContract = new Contract(MyContractAddress, MyContractABI, web3.getSigner());

const eventFilter = myContract.filters.MyEvent();

web3.on("block", async (blockNumber) => {
  console.log(`Block ${blockNumber} received.`);

  const events = await myContract.queryFilter(eventFilter, blockNumber - 10, blockNumber);

  for (const event of events) {
    console.log(`MyEvent emitted at block ${event.blockNumber}:`, event);

    // Take appropriate action based on the event data
    const { arg1, arg2 } = event;

    // Notify users of transactions
    console.log(`Notifying users of transaction with hash ${event.transactionHash}`);

    // Update a dashboard
    console.log(`Updating dashboard with event data: arg1=${arg1}, arg2=${arg2}`);
  }
});
