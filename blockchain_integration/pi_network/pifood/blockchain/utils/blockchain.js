const Web3 = require('web3');
const ethers = require('ethers');
const { abi, bytecode } = require('../contracts/InventoryManager.sol');

const BLOCKCHAIN_UTILS = {
  /**
   * Get the current block number
   * @returns {Promise<number>} The current block number
   */
  async getBlockNumber() {
    const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    return web3.eth.getBlockNumber();
  },

  /**
   * Get the current gas price
   * @returns {Promise<string>} The current gas price in wei
   */
  async getGasPrice() {
    const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
    return web3.eth.getGasPrice();
  },

  /**
   * Deploy a contract to the blockchain
   * @param {string} contractName The name of the contract to deploy
   * @param {string} bytecode The bytecode of the contract
   * @param {string} abi The ABI of the contract
   * @param {string} providerUrl The URL of the provider to use
   * @returns {Promise<string>} The address of the deployed contract
   */
  async deployContract(contractName, bytecode, abi, providerUrl) {
    const provider = new ethers.providers.JsonRpcProvider(providerUrl);
    const wallet = new ethers.Wallet('0x' + 'YOUR_PRIVATE_KEY');
    const contractFactory = new ethers.ContractFactory(abi, bytecode, wallet);
    const contract = await contractFactory.deploy();
    return contract.address;
  },

  /**
   * Get the balance of an address
   * @param {string} address The address to get the balance of
   * @param {string} providerUrl The URL of the provider to use
   * @returns {Promise<string>} The balance of the address in wei
   */
  async getBalance(address, providerUrl) {
    const provider = new ethers.providers.JsonRpcProvider(providerUrl);
    return provider.getBalance(address);
  },

  /**
   * Send a transaction to the blockchain
   * @param {string} from The address to send from
   * @param {string} to The address to send to
   * @param {string} value The value to send in wei
   * @param {string} providerUrl The URL of the provider to use
   * @returns {Promise<string>} The transaction hash
   */
  async sendTransaction(from, to, value, providerUrl) {
    const provider = new ethers.providers.JsonRpcProvider(providerUrl);
    const wallet = new ethers.Wallet('0x' + 'YOUR_PRIVATE_KEY');
    const tx = {
      from,
      to,
      value: ethers.utils.parseEther(value),
      gas: '20000',
      gasPrice: ethers.utils.parseUnits('20', 'gwei'),
    };
    return wallet.sendTransaction(tx);
  },
};

module.exports = BLOCKCHAIN_UTILS;
