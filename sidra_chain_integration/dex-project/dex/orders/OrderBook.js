import { AdvancedOrder } from './AdvancedOrder';
import { Web3Utils } from '../utils/Web3Utils';

class OrderBook {
  constructor(symbol) {
    this.symbol = symbol;
    this.bids = [];
    this.asks = [];
  }

  addOrder(order) {
    if (order.getSide() === 'buy') {
      this.bids.push(order);
      this.bids.sort((a, b) => b.getPrice().cmp(a.getPrice()));
    } else {
      this.asks.push(order);
      this.asks.sort((a, b) => a.getPrice().cmp(b.getPrice()));
    }
  }

  removeOrder(order) {
    if (order.getSide() === 'buy') {
      this.bids = this.bids.filter((o) =>!o.equals(order));
    } else {
      this.asks = this.asks.filter((o) =>!o.equals(order));
    }
  }

  getBestBid() {
    return this.bids[0];
  }

  getBestAsk() {
    return this.asks[0];
  }

  getOrders(side) {
    return side === 'buy'? this.bids : this.asks;
  }

  getSymbol() {
    return this.symbol;
  }

  toJSON() {
    return {
      symbol: this.symbol,
      bids: this.bids.map((o) => o.toJSON()),
      asks: this.asks.map((o) => o.toJSON()),
    };
  }

  static fromJSON(json) {
    const orderBook = new OrderBook(json.symbol);
    json.bids.forEach((o) => orderBook.addOrder(AdvancedOrder.fromJSON(o)));
    json.asks.forEach((o) => orderBook.addOrder(AdvancedOrder.fromJSON(o)));
    return orderBook;
  }
}

export default OrderBook;
