import { Web3 } from 'web3';
import { ethers } from 'ethers';
import { Polkadot } from 'polkadot-js';
import { InteroperabilityController } from './InteroperabilityController';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
const polkadot = new Polkadot('wss://rpc.polkadot.io');
const wallet = new ethers.Wallet('0x1234567890abcdef', web3);

const interoperabilityController = new InteroperabilityController(
  web3,
  polkadot,
  wallet
);

export default interoperabilityController;
