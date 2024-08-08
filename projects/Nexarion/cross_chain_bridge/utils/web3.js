import Web3 from 'web3';

const web3Ethereum = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
const web3BinanceSmartChain = new Web3(new Web3.providers.HttpProvider('https://bsc-dataseed.binance.org/api/v1/bc/BSC/main'));

export function getEthereumWeb3() {
  return web3Ethereum;
}

export function getBinanceSmartChainWeb3() {
  return web3BinanceSmartChain;
}

export function getAccountBalance(web3, address) {
  return web3.eth.getBalance(address);
}

export function sendTransaction(web3, from, to, value, gas, gasPrice) {
  return web3.eth.sendTransaction({
    from,
    to,
    value,
    gas,
    gasPrice
  });
}
