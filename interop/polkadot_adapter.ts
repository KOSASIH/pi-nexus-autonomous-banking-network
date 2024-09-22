import { ApiPromise } from '@polkadot/api';

class PolkadotAdapter {
  private api: ApiPromise;

  constructor() {
    this.api = new ApiPromise();
  }

  async sendTransaction(transaction: any) {
    return this.api.tx.transaction(transaction);
  }

  async receiveTransaction(transaction: any) {
    return this.api.query.transactionStatus(transaction.hash);
  }
}

export default PolkadotAdapter;
