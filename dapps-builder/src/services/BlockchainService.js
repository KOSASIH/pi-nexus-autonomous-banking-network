import web3 from '../components/BlockchainIntegration/web3';
import { getContractInstance, callContractMethod, sendTransaction } from '../components/BlockchainIntegration/contractInteraction';

class BlockchainService {
    constructor(abi, contractAddress) {
        this.contract = getContractInstance(abi, contractAddress);
    }

    async getAccounts() {
        return await web3.eth.getAccounts();
    }

    async getNetworkId() {
        return await web3.eth.net.getId();
    }

    async callMethod(method, args) {
        return await callContractMethod(this.contract, method, args);
    }

    async sendMethod(method, args, from) {
        return await sendTransaction(this.contract, method, args, from);
    }

    async getTransactionReceipt(txHash) {
        return await web3.eth.getTransactionReceipt(txHash);
    }
}

export default BlockchainService;
