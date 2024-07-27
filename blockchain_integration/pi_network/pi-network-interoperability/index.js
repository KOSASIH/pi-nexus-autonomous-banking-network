const RelayerNode = require('./RelayerNode');
const Web3Utils = require('./utils/Web3');
const EthersUtils = require('./utils/Ethers');

const bridgeContractAddress = '0x...';
const crossChainMessageContractAddress = '0x...';
const atomicSwapContractAddress = '0x...';

const relayerNode = new RelayerNode(bridgeContractAddress, crossChainMessageContractAddress, atomicSwapContractAddress);

async function main() {
  // Bridge token example
  const tokenAddress = '0x...';
  const recipientAddress = '0x...';
  const amount = 1;

  const bridgedToken = await relayerNode.bridgeToken(tokenAddress, recipientAddress, amount);
  console.log(`Bridged token: ${bridgedToken}`);

  // Send message example
  const messageId = '0x...';
  const message = 'Hello, cross-chain!';

  const sentMessage = await relayerNode.sendMessage(messageId, message);
  console.log(`Sent message: ${sentMessage}`);

  // Execute swap example
  const swapId = '0x...';
  const senderAddress = '0x...';
  const recipientAddress = '0x...';
  const amount = 1;

  const executedSwap = await relayerNode.executeSwap(swapId, senderAddress, recipientAddress, amount);
  console.log(`Executed swap: ${executedSwap}`);
}

main();
