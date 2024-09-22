import * as crypto from 'crypto';

interface PublicKey {
  n: number;
  q: number;
  g: number;
}

interface PrivateKey {
  p: number;
  q: number;
  d: number;
}

class NTRU {
  private publicKey: PublicKey;
  private privateKey: PrivateKey;

  constructor(publicKey: PublicKey, privateKey: PrivateKey) {
    this.publicKey = publicKey;
    this.privateKey = privateKey;
  }

  async encrypt(message: string): Promise<string> {
    const m = this.messageToNumber(message);
    const r = this.generateRandomNumber();
    const c = (this.publicKey.g * r + m) % this.publicKey.q;
    return this.numberToString(c);
  }

  async decrypt(ciphertext: string): Promise<string> {
    const c = this.stringToNumber(ciphertext);
    const m = (c * this.privateKey.d) % this.privateKey.q;
    return this.numberToString(m);
  }

  private messageToNumber(message: string): number {
    const hash = crypto.createHash('sha256');
    hash.update(message);
    return parseInt(hash.digest('hex'), 16);
  }

  private numberToString(number: number): string {
    return number.toString(16);
  }

  private stringToNumber(string: string): number {
    return parseInt(string, 16);
  }

  private generateRandomNumber(): number {
    return Math.floor(Math.random() * this.publicKey.q);
  }
}

export default NTRU;
