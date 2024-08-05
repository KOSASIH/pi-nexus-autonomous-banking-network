const HDWalletProvider = require("truffle-hdwallet-provider");
const Web3 = require("web3");
const { abi, bytecode } = require("../contracts/TradeFinance.json");
const { abi: tokenAbi, bytecode: tokenBytecode } = require("../contracts/PiTradeToken.json");

const provider = new HDWalletProvider(
  "0x1234567890abcdef", // Your Ethereum wallet private key
  "https://mainnet.infura.io/v3/YOUR_PROJECT_ID" // Your Infura project ID
);

const web3 = new Web3(provider);

async function deploy() {
  const accounts = await web3.eth.getAccounts();
  const deployer = accounts[0];

  console.log("Deploying TradeFinance contract...");
  const tradeFinanceContract = new web3.eth.Contract(abi);
  const tradeFinanceTx = tradeFinanceContract.deploy({ data: bytecode });
  const tradeFinanceReceipt = await tradeFinanceTx.send({ from: deployer, gas: 5000000 });
  const tradeFinanceAddress = tradeFinanceReceipt.contractAddress;
  console.log(`TradeFinance contract deployed to ${tradeFinanceAddress}`);

  console.log("Deploying PiTradeToken contract...");
  const piTradeTokenContract = new web3.eth.Contract(tokenAbi);
  const piTradeTokenTx = piTradeTokenContract.deploy({ data: tokenBytecode });
  const piTradeTokenReceipt = await piTradeTokenTx.send({ from: deployer, gas: 5000000 });
  const piTradeTokenAddress = piTradeTokenReceipt.contractAddress;
  console.log(`PiTradeToken contract deployed to ${piTradeTokenAddress}`);

  console.log("Setting PiTradeToken address in TradeFinance contract...");
  const tradeFinanceInstance = new web3.eth.Contract(abi, tradeFinanceAddress);
  const setPiTradeTokenTx = tradeFinanceInstance.methods.setPiTradeToken(piTradeTokenAddress).send({ from: deployer, gas: 2000000 });
  await setPiTradeTokenTx;
  console.log("PiTradeToken address set in TradeFinance contract");
}

deploy();
