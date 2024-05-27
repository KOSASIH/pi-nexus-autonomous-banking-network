import Web3 from "web3";
import { Contract } from "web3/eth/contract";
import DEXFactoryABI from "./abis/DEXFactory.json";
import DEXABI from "./abis/DEX.json";

class DEX {
  constructor() {
    this.web3 = new Web3(
      new Web3.providers.HttpProvider(
        "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
      ),
    );
    this.dexFactoryContract = new Contract(
      "0x1F98431c8aD98523631AE4a59f267346ea31F984", // DEXFactory contract address
      DEXFactoryABI,
      this.web3.currentProvider,
    );
  }

  async createDEX(tokenA, tokenB, fee) {
    const poolAddress = await this.dexFactoryContract.methods
      .getPool(tokenA, tokenB, fee)
      .call();
    const dexContract = new Contract(
      poolAddress,
      DEXABI,
      this.web3.currentProvider,
    );
    return dexContract;
  }
}

export default DEX;
