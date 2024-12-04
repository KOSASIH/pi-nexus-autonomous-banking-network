// smartContracts.js

import { ethers } from 'ethers';

class SmartContractManager {
    constructor(providerUrl, privateKey) {
        this.provider = new ethers.providers.JsonRpcProvider(providerUrl);
        this.wallet = new ethers.Wallet(privateKey, this.provider);
    }

    // Deploy a new smart contract
    async deployContract(abi, bytecode, constructorArgs) {
        const factory = new ethers.ContractFactory(abi, bytecode, this.wallet);
        const contract = await factory.deploy(...constructorArgs);
        await contract.deployed();
        console.log('Contract deployed at address:', contract.address);
        return contract.address;
    }

    // Interact with an existing smart contract
    async interactWithContract(contractAddress, abi, methodName, methodArgs) {
        const contract = new ethers.Contract(contractAddress, abi, this.wallet);
        const tx = await contract[methodName](...methodArgs);
        await tx.wait();
        console.log(`Transaction successful: ${tx.hash}`);
        return tx;
    }

    // Get the balance of a smart contract
    async getContractBalance(contractAddress) {
        const balance = await this.provider.getBalance(contractAddress);
        console.log(`Contract balance: ${ethers.utils.formatEther(balance)} ETH`);
        return balance;
    }

    // Example usage
    async exampleUsage() {
        const abi = [
            // Example ABI for a simple storage contract
            "function set(uint x) public",
            "function get() public view returns (uint)"
        ];
        const bytecode = "0x..."; // Replace with actual bytecode

        // Deploy a new contract
        const contractAddress = await this.deployContract(abi, bytecode, []);

        // Interact with the deployed contract
        await this.interactWithContract(contractAddress, abi, 'set', [42]);
        const value = await this.interactWithContract(contractAddress, abi, 'get', []);
        console.log('Stored value:', value.toString());

        // Get contract balance
        await this.getContractBalance(contractAddress);
    }
}

// Example usage
const providerUrl = 'https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID'; // Replace with your provider URL
const privateKey = 'YOUR_PRIVATE_KEY'; // Replace with your wallet's private key
const smartContractManager = new SmartContractManager(providerUrl, privateKey);
smartContractManager.exampleUsage();

export default SmartContractManager;
