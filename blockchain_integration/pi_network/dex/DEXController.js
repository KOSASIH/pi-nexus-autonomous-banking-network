import DEX from './DEX';
import { ethers } from 'ethers';

class DEXController {
  constructor(dex) {
    this.dex = dex;
  }

  async getPool(tokenA, tokenB, fee) {
    const dexContract = await this.dex.createDEX(tokenA, tokenB, fee);
    return dexContract;
  }

  async getPrice(tokenA, tokenB, fee) {
    const dexContract = await this.getPool(tokenA, tokenB, fee);
    const slot0 = await dexContract.methods.slot0().call();
    const tick = slot0.tick;
    const tokenBPrice = ethers.utils.parseUnits('1', tokenA.decimals).div(
      ethers.utils.parseUnits(tick.toString(), 27).mul(10**tokenB.decimals)
    );
    return tokenBPrice.toString();
  }
}

export default DEXController;
