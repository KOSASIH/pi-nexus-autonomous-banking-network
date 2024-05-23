// interfaces/pi-rpc.js

import { EventEmitter } from 'events';
import { WebSocket } from 'ws';
import { abi } from './pi-rpc.abi';

const PI_NETWORK_RPC_VERSION = '2.0';
const PI_NETWORK_RPC_ID = 1;

interface PIRPCRequest {
  jsonrpc: string;
  method: string;
  params: any[];
  id: number;
}

interface PIRPCResponse {
  jsonrpc: string;
  id: number;
  result?: any;
  error?: {
    code: number;
    message: string;
  };
}

export class PIRPC extends EventEmitter {
  private rpcId = 1;
  private ws: WebSocket;
  private pendingRequests: { [id: number]: (response: PIRPCResponse) => void } = {};

  constructor(url: string) {
    super();

    this.ws = new WebSocket(url);

    this.ws.on('open', () => {
      this.emit('connect');
    });

    this.ws.on('message', (data: string) => {
      const response: PIRPCResponse = JSON.parse(data);
      const handler = this.pendingRequests[response.id];

      if (handler) {
        handler(response);
        delete this.pendingRequests[response.id];
      }
    });

    this.ws.on('close', () => {
      this.emit('disconnect');
    });

    this.ws.on('error', (error) => {
      this.emit('error', error);
    });
  }

  async call(method: string, params: any[]): Promise<any> {
    const request: PIRPCRequest = {
      jsonrpc: PI_NETWORK_RPC_VERSION,
      method,
      params,
      id: this.rpcId++
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests[request.id] = (response: PIRPCResponse) => {
        if (response.error) {
          reject(response.error);
        } else {
          resolve(response.result);
        }
      };

      this.ws.send(JSON.stringify(request));
    });
  }

  async getBlockNumber(): Promise<number> {
    return this.call('pi_getBlockNumber', []);
  }

  async getBlockByNumber(blockNumber: number): Promise<any> {
    return this.call('pi_getBlockByNumber', [blockNumber]);
  }

  async getTransactionCount(address: string): Promise<number> {
    return this.call('pi_getTransactionCount', [address]);
  }

  async getTransactionByHash(hash: string): Promise<any> {
    return this.call('pi_getTransactionByHash', [hash]);
  }

  async sendTransaction(tx: any): Promise<string> {
    return this.call('pi_sendTransaction', [tx]);
  }

  async getBalance(address: string): Promise<string> {
    return this.call('pi_getBalance', [address]);
  }

  async getStorageAt(address: string, position: number): Promise<string> {
    return this.call('pi_getStorageAt', [address, position]);
  }

  async getLogs(filter: any): Promise<any[]> {
    return this.call('pi_getLogs', [filter]);
  }
}

export default PIRPC;
