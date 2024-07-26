import Web3 from 'web3';
import { DEXContract } from './contracts/DEXContract';

class OrderManager {
  constructor(dexContract) {
    this.dexContract = dexContract;
    this.orders = {};
  }

  async placeOrder(amount, price, side) {
    const tx = await this.dexContract.placeOrder(amount, price, side);
    const orderId = tx.events.OrderPlaced.returnValues.orderId;

    this.orders[orderId] = {
      amount: amount,
      price: price,
      side: side,
    };

    console.log(`Order placed: ${orderId} - ${amount} ${side === 0? 'buy' : 'ell'} @ ${price}`);
  }

  async cancelOrder(orderId) {
    const tx = await this.dexContract.cancelOrder(orderId);
    delete this.orders[orderId];

    console.log(`Order cancelled: ${orderId}`);
  }

  async getOrders() {
    return this.orders;
  }
}

export { OrderManager };
