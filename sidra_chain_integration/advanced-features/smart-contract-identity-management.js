const Web3 = require('web3');

async function createIdentityContract(userInput) {
  const web3 = new Web3(new Web3.providers.HttpProvider('https://sidra-chain-node.com'));
  const contract = new web3.eth.Contract([
    {
      constant: true,
      inputs: [],
      name: 'createIdentity',
      outputs: [{ name: '', type: 'address' }],
      payable: false,
      stateMutability: 'nonpayable',
      type: 'function'
    }
  ], '0x...SidraChainContractAddress...');
  const result = await contract.methods.createIdentity(userInput).send();
  return result;
}

module.exports = { createIdentityContract };
