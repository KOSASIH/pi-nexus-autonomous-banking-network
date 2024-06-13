import { DistributedLedgerTechnology } from 'distributed-ledger-technology-sdk';

class DistributedLedgerTechnology {
  constructor() {
    this.distributedLedgerTechnology = new DistributedLedgerTechnology();
  }

  async createDistributedLedger(ledgerConfig) {
    const ledger = await this.distributedLedgerTechnology.create(ledgerConfig);
    return ledger;
  }
}

export default DistributedLedgerTechnology;
