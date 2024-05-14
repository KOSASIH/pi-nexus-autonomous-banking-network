const { Contract, providers } = require('ethers');
const { ChainlinkClient } = require('@chainlink/external-adapter-framework');
const { JsonRpcProvider } = require('@ethersproject/providers');

const provider = new JsonRpcProvider('https://eth-mainnet.alchemyapi.io/v2/your-alchemy-api-key');
const contractAddress = '0xYourContractAddress';
const contractAbi = [
  // Add the ABI of your ERC-725 contract here
];
const contract = new Contract(contractAddress, contractAbi, provider);

const chainlinkClient = new ChainlinkClient({
  provider: provider,
  contractAddress: contractAddress,
});

const sendCrossChainMessage = async (destinationChain, message) => {
  const result = await chainlinkClient.request({
    chainId: 1, // Ethereum mainnet chain ID
    operationName: 'sendCrossChainMessage',
    params: {
      destinationChain: destinationChain,
      message: message,
    },
  });

  console.log('Cross-chain message sent:', result);
};

// Example usage
const destinationChain = 1234; // Replace with the ID of the destination chain
const message = 'Hello, cross-chain world!';

sendCrossChainMessage(destinationChain, message);
