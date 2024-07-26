import { SidraChain } from '../sidra-chain';
import { IPFS } from 'ipfs-js';

class DDS {
  constructor(sidraChain) {
    this.sidraChain = sidraChain;
    this.ipfs = new IPFS();
  }

  async storeData(data) {
    // Decentralized data storage
    const cid = await this.ipfs.add(data);
    return cid;
  }
}

export { DDS };
