import { Web3 } from 'web3';

class SmartContractInteraction {
  constructor() {
    this.web3 = new Web3();
  }

  async interactWithContract(contractAddress, abi, functionCall) {
    const contract = await this.web3.eth.Contract(contractAddress, abi);
    const result = await contract.methods[functionCall]().call();
    return result;
  }
}

export default SmartContractInteraction;
