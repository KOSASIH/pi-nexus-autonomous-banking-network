const { ethers } = require("hardhat");
const { PiTradeToken } = require("../contracts/PiTradeToken.sol");
const { TradeFinance } = require("../contracts/TradeFinance.sol");

async function deploy() {
  // Set up the provider and signer
  const provider = new ethers.providers.JsonRpcProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID");
  const signer = new ethers.Wallet("0xYOUR_PRIVATE_KEY", provider);

  // Deploy the PiTradeToken contract
  const piTradeTokenFactory = new ethers.ContractFactory(PiTradeToken.abi, PiTradeToken.bytecode, signer);
  const piTradeToken = await piTradeTokenFactory.deploy();
  await piTradeToken.deployed();
  console.log(`PiTradeToken deployed to ${piTradeToken.address}`);

  // Deploy the TradeFinance contract
  const tradeFinanceFactory = new ethers.ContractFactory(TradeFinance.abi, TradeFinance.bytecode, signer);
  const tradeFinance = await tradeFinanceFactory.deploy(piTradeToken.address);
  await tradeFinance.deployed();
  console.log(`TradeFinance deployed to ${tradeFinance.address}`);

  // Set up the TradeFinance contract with the PiTradeToken contract
  await tradeFinance.setPiTradeToken(piTradeToken.address);
  console.log("TradeFinance contract set up with PiTradeToken contract");

  // Verify the contracts on Etherscan
  await verifyContract(piTradeToken.address, PiTradeToken.abi, PiTradeToken.bytecode);
  await verifyContract(tradeFinance.address, TradeFinance.abi, TradeFinance.bytecode);
  console.log("Contracts verified on Etherscan");
}

async function verifyContract(address, abi, bytecode) {
  const etherscanApiUrl = "https://api.etherscan.io/api";
  const apiKey = "YOUR_ETHERSCAN_API_KEY";
  const params = {
    module: "contract",
    action: "verifysourcecode",
    contractaddress: address,
    sourceCode: bytecode,
    contractname: abi.contractName,
    compilerversion: "v0.8.0",
    optimizationUsed: 1,
    runs: 200,
    constructorArguments: "",
    licenseType: 1,
    apikey: apiKey,
  };

  const response = await fetch(`${etherscanApiUrl}?${new URLSearchParams(params)}`);
  const data = await response.json();
  if (data.status === "1") {
    console.log(`Contract verified on Etherscan: ${address}`);
  } else {
    console.error(`Error verifying contract on Etherscan: ${address}`);
  }
}

deploy();
