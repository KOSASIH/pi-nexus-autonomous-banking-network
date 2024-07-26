import { SidraChain } from '../sidra-chain';
import { LatticeCrypto } from 'lattice-crypto-js';

class QRDS {
  constructor(sidraChain) {
    this.sidraChain = sidraChain;
    this.latticeCrypto = new LatticeCrypto();
  }

  async signTransaction(transaction) {
    // Quantum-resistant digital signature
    const signature = await this.latticeCrypto.sign(transaction);
    return signature;
  }
}

export { QRDS };
