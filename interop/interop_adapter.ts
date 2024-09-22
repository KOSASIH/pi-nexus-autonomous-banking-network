import { EthereumAdapter } from './ethereum_adapter';
import { PolkadotAdapter } from './polkadot_adapter';

interface BlockchainNetwork {
  name: string;
  adapter: any;
}

class InteropAdapter {
  private networks: BlockchainNetwork[];

  constructor() {
    this.networks = [
      { name: 'Ethereum', adapter: new EthereumAdapter() },
      { name: 'Polkadot', adapter: new PolkadotAdapter() },
    ];
  }

  async sendTransaction(network: string, transaction: any) {
    const adapter = this.getAdapter(network);
    if (!adapter) {
      throw new Error(`Unsupported network: ${network}`);
    }
    return adapter.sendTransaction(transaction);
  }

  async receiveTransaction(network: string, transaction: any) {
    const adapter = this.getAdapter(network);
    if (!adapter) {
      throw new Error(`Unsupported network: ${network}`);
    }
    return adapter.receiveTransaction(transaction);
  }

  private getAdapter(network: string) {
    return this.networks.find((n) => n.name === network)?.adapter;
  }
}

export default InteropAdapter;
