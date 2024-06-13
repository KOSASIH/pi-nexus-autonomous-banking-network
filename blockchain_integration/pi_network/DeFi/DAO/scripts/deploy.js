const Web3 = require("web3");
const DAO = require("./DAO.sol");
const Governance = require("./Governance.sol");
const Token = require("./Token.sol");

async function deploy() {
    // Set up Web3 provider
    const web3 = new Web3(new Web3.providers.HttpProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"));

    // Deploy DAO contract
    const dao = await deployContract(web3, DAO, "DAO", "DAO Token");

    // Deploy Governance contract
    const governance = await deployContract(web3, Governance, "Governance", "Governance Contract");

    // Deploy Token contract
    const token = await deployContract(web3, Token, "Token", "Token Contract");

    // Set up DAO contract
    await dao.methods.initialize(governance.address, token.address).send({ from: "0xYourAddress" });

    // Set up Governance contract
    await governance.methods.initialize(dao.address).send({ from: "0xYourAddress" });

    // Set up Token contract
   await token.methods.initialize(dao.address).send({ from: "0xYourAddress" });

    console.log("DAO contract address:", dao.options.address);
    console.log("Governance contract address:", governance.options.address);
    console.log("Token contract address:", token.options.address);
}

async function deployContract(web3, contract, name, symbol) {
    // Compile the contract
    const compiledContract = await web3.eth.compile(contract);

    // Deploy the contract
    const deployTx = await web3.eth.sendTransaction({
        from: "0xYourAddress",
        data: compiledContract.bytecode,
        gas: "2000000",
        gasPrice: web3.utils.toWei("20", "gwei")
    });

    // Get the contract address
    const contractAddress = await web3.eth.getTransactionReceipt(deployTx.transactionHash);

    // Return the contract instance
    return new web3.eth.Contract(compiledContract.abi, contractAddress.contractAddress);
}

deploy();
