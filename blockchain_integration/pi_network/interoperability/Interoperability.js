import { Web3 } from "web3";
import { ethers } from "ethers";
import { Polkadot } from "polkadot-js";
import { InteroperabilityController } from "./InteroperabilityController";

class Interoperability {
  constructor() {
    this.web3 = new Web3(
      new Web3.providers.HttpProvider(
        "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
      ),
    );
    this.ethers = ethers;
    this.polkadot = new Polkadot("wss://rpc.polkadot.io");
    this.controller = new InteroperabilityController(
      this.web3,
      this.ethers,
      this.polkadot,
    );
  }

  async sendTransactionToPolkadot(transaction) {
    return await this.controller.sendTransactionToPolkadot(transaction);
  }

  async sendTransactionToEthereum(transaction) {
    return await this.controller.sendTransactionToEthereum(transaction);
  }

  async getBalanceOnPolkadot(address) {
    return await this.controller.getBalanceOnPolkadot(address);
  }

  async getBalanceOnEthereum(address) {
    return await this.controller.getBalanceOnEthereum(address);
  }
}

export default Interoperability;
