import { Web3 } from 'web3';
import { ethers } from 'ethers';
import { CoinJoin } from 'coinjoin';
import { ConfidentialTransactions } from 'confidential-transactions';
import { zkSNARKs } from 'zk-snarks';
import { TransactionPrivacyController } from './TransactionPrivacyController';

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
const wallet = new ethers.Wallet('0x1234567890abcdef', web3);
const coinJoinContract = new CoinJoin('0xCOINJOIN_CONTRACT_ADDRESS', web3);
const confidentialTransactionsContract = new ConfidentialTransactions('0xCONFIDENTIAL_TRANSACTIONS_CONTRACT_ADDRESS', web3);
const zkSnarks = new zkSNARKs();

const transactionPrivacyController = new TransactionPrivacyController(
  wallet,
  coinJoinContract,
  confidentialTransactionsContract,
  zkSnarks
);

export default transactionPrivacyController;
