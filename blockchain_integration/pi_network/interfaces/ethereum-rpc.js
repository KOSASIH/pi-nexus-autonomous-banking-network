// interfaces/ethereum-rpc.js

import { EventEmitter } from 'events';
import { WebSocket } from 'ws';
import { ethers } from 'ethers';
import { abi } from './ethereum-rpc.abi';

const ETHEREUM_RPC_VERSION = '2.0';
const ETHEREUM_RPC_ID = 1;

interface EthereumRPCRequest {
  jsonrpc: string;
  method: string;
  params: any[];
  id: number;
}

interface EthereumRPCResponse {
  jsonrpc: string;
  result: any;
  error: any;
  id: number;
}

class EthereumRPC extends EventEmitter {
  private ws: WebSocket;
  private rpcId: number;
  private pendingRequests: { [id: number]: (response: EthereumRPCResponse) => void };

  constructor(url: string) {
    super();
    this.ws = new WebSocket(url);
    this.rpcId = ETHEREUM_RPC_ID;
    this.pendingRequests = {};

    this.ws.on('open', () => {
      this.emit('connected');
    });

    this.ws.on('message', (data: string) => {
      const response: EthereumRPCResponse = JSON.parse(data);
      if (response.id in this.pendingRequests) {
        this.pendingRequests[response.id](response);
        delete this.pendingRequests[response.id];
      } else {
        this.emit('response', response);
      }
    });

    this.ws.on('error', (error: Error) => {
      this.emit('error', error);
    });

    this.ws.on('close', () => {
      this.emit('disconnected');
    });
  }

  async call(method: string, params: any[]): Promise<any> {
    const request: EthereumRPCRequest = {
      jsonrpc: ETHEREUM_RPC_VERSION,
      method,
      params,
      id: this.rpcId++
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests[request.id] = (response: EthereumRPCResponse) => {
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
    return this.call('eth_blockNumber', []);
  }

  async getBlockByNumber(blockNumber: number): Promise<any> {
    return this.call('eth_getBlockByNumber', [ethers.utils.hexlify(blockNumber)]);
  }

  async getTransactionCount(address: string): Promise<number> {
    return this.call('eth_getTransactionCount', [address]);
  }

  async getTransactionByHash(hash: string): Promise<any> {
    return this.call('eth_getTransactionByHash', [hash]);
  }

  async sendTransaction(tx: any): Promise<string> {
    return this.call('eth_sendTransaction', [tx]);
  }

  async getBalance(address: string): Promise<string> {
    return this.call('eth_getBalance', [address]);
  }

  async getStorageAt(address: string, position: number): Promise<string> {
    return this.call('eth_getStorageAt', [address, ethers.utils.hexlify(position)]);
  }

  async getLogs(filter: any): Promise<any[]> {
    return this.call('eth_getLogs', [filter]);
  }
}

export default EthereumRPC;
