import { Injectable } from '@nestjs/common';
import { WebSocket } from 'ws';
import { Trade } from './Trade';

@Injectable()
export class TradeService {
  private ws: WebSocket;
  private trades: Trade[] = [];

  constructor() {
    this.ws = new WebSocket('wss://dex.example.com/trades');
  }

  async init() {
    this.ws.on('message', (data) => {
      const trade = JSON.parse(data);
      this.trades.push(trade);
    });
  }

  async getTrades() {
    return this.trades;
  }

  async placeTrade(trade: Trade) {
    this.ws.send(JSON.stringify(trade));
  }
}
