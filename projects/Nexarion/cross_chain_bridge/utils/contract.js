import { getEthereumWeb3, getBinanceSmartChainWeb3 } from './web3';

export function getContract(web3, abi, address) {
  return new web3.eth.Contract(abi, address);
}

export function getEthereumContract(abi, address) {
  return getContract(getEthereumWeb3(), abi, address);
}

export function getBinanceSmartChainContract(abi, address) {
  return getContract(getBinanceSmartChainWeb3(), abi, address);
}
