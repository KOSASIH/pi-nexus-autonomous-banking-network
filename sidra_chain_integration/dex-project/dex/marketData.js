import Web3 from 'web3';
import { TokenContract } from './contracts/TokenContract';
import { DEXContract } from './contracts/DEXContract';

class MarketData {
  constructor(dexContract, tokenContract) {
    this.dexContract = dexContract;
    this.tokenContract = tokenContract;
    this.marketData = {};
  }

  async start() {
    this.updateMarketData();
    setInterval(this.updateMarketData.bind(this), 10000);
  }

  async updateMarketData() {
    const orders = await this.dexContract.getOrders();
    const tokenBalance = await this.tokenContract.balanceOf(this.dexContract.address);

    this.marketData = {
      orders: orders.map((order) => ({
        id: order.id,
        amount: order.amount,
        price: order.price,
        side: order.side,
      })),
      tokenBalance: tokenBalance,
    };

    console.log('Market data updated:', this.marketData);
  }

  async getMarketData() {
    return this.marketData;
  }
}

export { MarketData };
