import { Injectable } from '@nestjs/common';
import { WebSocket } from 'ws';
import { OrderBook } from './OrderBook';

@Injectable()
export class OrderBookService {
  private orderBook: OrderBook;
  private ws: WebSocket;

  constructor() {
    this.orderBook = new OrderBook();
    this.ws = new WebSocket('wss://dex.example.com/orderbook');
  }

  async init() {
    this.ws.on('message', (data) => {
      const message = JSON.parse(data);
      switch (message.type) {
        case 'orderBook':
          this.orderBook.update(message.data);
          break;
        case 'order':
          this.orderBook.addOrder(message.data);
          break;
        case 'cancel':
          this.orderBook.cancelOrder(message.data);
          break;
      }
    });
  }

  async getOrderBook() {
    return this.orderBook.getSnapshot();
  }

  async placeOrder(order: any) {
    this.ws.send(JSON.stringify({ type: 'order', data: order }));
  }

  async cancelOrder(orderId: string) {
    this.ws.send(JSON.stringify({ type: 'cancel', data: orderId }));
  }
      }
