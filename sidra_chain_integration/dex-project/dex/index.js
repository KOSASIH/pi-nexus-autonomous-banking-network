import Web3 from 'web3';
import { SidraChainSDK } from 'sidra-chain-sdk';
import { DEXContract } from './contracts/DEXContract';
import { TokenContract } from './contracts/TokenContract';
import { OrderBook } from './orderBook';
import { MarketData } from './marketData';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.sidra.chain'));
const sidraChainSDK = new SidraChainSDK(web3);

const dexContract = new DEXContract(sidraChainSDK, '0x...DEXContractAddress...');
const tokenContract = new TokenContract(sidraChainSDK, '0x...TokenContractAddress...');

const orderBook = new OrderBook(dexContract);
const marketData = new MarketData(dexContract, tokenContract);

async function start() {
  await dexContract.deployed();
  await tokenContract.deployed();

  console.log('DEX Contract deployed at:', dexContract.address);
  console.log('Token Contract deployed at:', tokenContract.address);

  orderBook.start();
  marketData.start();

  console.log('Sidra Chain DEX started!');
}

start();
