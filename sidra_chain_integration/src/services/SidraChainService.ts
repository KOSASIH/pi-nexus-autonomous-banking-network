import { Injectable } from '@nestjs/common';
import { SidraChain } from '../contracts/SidraChain';
import Web3 from 'web3';

@Injectable()
export class SidraChainService {
  private web3: Web3;
  private sidraChainContract: SidraChain;

  constructor() {
    this.web3 = new Web3(new Web3.providers.HttpProvider(environment.sidraChainUrl));
    this.sidraChainContract = new SidraChain(environment.sidraChainContractAddress);
  }

  async getSidraChainInfo() {
    const info = await this.sidraChainContract.methods.getInfo().call();
    return info;
  }

  async transfer(transferData: TransferData) {
    const txCount = await this.web3.eth.getTransactionCount(environment.tokenContractAddress);
    const tx = {
      from: environment.tokenContractAddress,
      to: transferData.recipient,
      value: transferData.amount,
      gas: '20000',
      gasPrice: '20',
      nonce: txCount,
    };
    const signedTx = await this.web3.eth.accounts.signTransaction(tx, environment.privateKey);
    const receipt = await this.web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    return receipt;
  }

  //...
}
