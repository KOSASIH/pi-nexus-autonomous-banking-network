import { NTRU } from 'ntru-js';

class QuantumCrypto {
  private ntru: NTRU;

  constructor() {
    this.ntru = new NTRU();
  }

  async encrypt(data: string): Promise<string> {
    const publicKey = await this.ntru.generateKeyPair();
    const encryptedData = await this.ntru.encrypt(data, publicKey);
    return encryptedData;
  }

  async decrypt(encryptedData: string): Promise<string> {
    const privateKey = await this.ntru.generateKeyPair();
    const decryptedData = await this.ntru.decrypt(encryptedData, privateKey);
    return decryptedData;
  }
}

export default QuantumCrypto;
