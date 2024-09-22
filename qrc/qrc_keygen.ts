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

class KeyGenerator {
  async generateKeys(): Promise<{ publicKey: PublicKey; privateKey: PrivateKey }> {
    const p = this.generatePrimeNumber();
    const q = this.generatePrimeNumber();
    const g = this.generateRandomNumber(p);
    const d = this.generateRandomNumber(q);
    const publicKey: PublicKey = { n: p * q, q, g };
    const privateKey: PrivateKey = { p, q, d };
    return { publicKey, privateKey };
  }

  private generatePrimeNumber(): number {
    let p = Math.floor(Math.random() * 1000) + 1;
    while (!this.isPrime(p)) {
      p = Math.floor(Math.random() * 1000) + 1;
    }
    return p;
  }

  private isPrime(n: number): boolean {
    if (n <= 1) return false;
    if (n === 2) return true;
    if (n % 2 === 0) return false;
    for (let i = 3; i * i <= n; i += 2) {
      if (n % i === 0) return false;
    }
    return true;
  }

  private generateRandomNumber(max: number): number {
    return Math.floor(Math.random() * max);
  }
}

export default KeyGenerator;
