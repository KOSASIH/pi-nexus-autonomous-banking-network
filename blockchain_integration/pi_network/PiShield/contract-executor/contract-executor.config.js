// contract-executor.config.js

module.exports = {
  // Provider URL
  providerUrl: 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID',

  // Contract address and ABI
  contractAddress: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
  abi: [
    {
      "constant": true,
      "inputs": [],
      "name": "name",
      "outputs": [
        {
          "name": "",
          "type": "string"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "name": "_from",
          "type": "address"
        },
        {
          "name": "_to",
          "type": "address"
        },
        {
          "name": "_value",
          "type": "uint256"
        }
      ],
      "name": "transfer",
      "outputs": [
        {
          "name": "",
          "type": "bool"
        }
      ],
      "payable": false,
      "stateMutability": "nonpayable",
      "type": "function"
    }
  ],

  // From address and private key
  fromAddress: '0x1234567890abcdef1234567890abcdef',
  privateKey: '0x1234567890abcdef1234567890abcdef1234567890abcdef',

  // Gas and gas price
  gas: 200000,
  gasPrice: 20e9
};
