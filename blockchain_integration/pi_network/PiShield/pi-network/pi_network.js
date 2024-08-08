// pi_network.js

const { Node } = require('./node');
const { Network } = require('./network');
const { Blockchain } = require('./blockchain');
const { Logger } = require('./logger');

class PiNetwork {
  constructor(config) {
    this.config = config;
    this.nodes = [];
    this.network = new Network(config.network);
    this.blockchain = new Blockchain(config.blockchain);
    this.logger = new Logger(config.logger);
  }

  async init() {
    await this.network.init();
    await this.blockchain.init();
    for (let i = 0; i < this.config.nodes; i++) {
      const node = new Node(this.config.node);
      await node.init();
      this.nodes.push(node);
    }
  }

  async start() {
    await this.network.start();
    await this.blockchain.start();
    for (const node of this.nodes) {
      await node.start();
    }
  }

  async stop() {
    await this.network.stop();
    await this.blockchain.stop();
    for (const node of this.nodes) {
      await node.stop();
    }
  }

  async addNode(nodeConfig) {
    const node = new Node(nodeConfig);
    await node.init();
    this.nodes.push(node);
  }

  async removeNode(nodeId) {
    const node = this.nodes.find((node) => node.id === nodeId);
    if (node) {
      await node.stop();
      this.nodes = this.nodes.filter((node) => node.id !== nodeId);
    }
  }
}

module.exports = PiNetwork;
