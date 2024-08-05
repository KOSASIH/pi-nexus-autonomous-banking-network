const Migrations = require("truffle-migrations");
const Web3 = require("web3");

const provider = new Web3.providers.HttpProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"); // Your Infura project ID

const migrations = new Migrations(provider);

async function migrate() {
  console.log("Running migrations...");
  await migrations.run();
  console.log("Migrations complete");
}

migrate();
