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

class QRCUtils {
  async encrypt(message: string, publicKey: PublicKey): Promise<string> {
    const ntru = new NTRU(publicKey, null);
    return ntru.encrypt(message);
  }

  async decrypt(ciphertext: string, privateKey: PrivateKey): Promise<string> {
    const ntru = new NTRU(null, privateKey);
    return ntru.decrypt(ciphertext);
  }

  async sign(message: string, privateKey: PrivateKey): Promise<string> {
    const hash = crypto.createHash('sha256');
    hash.update(message);
    const signature = this.signHash(hash.digest('hex'), privateKey);
    return signature;
  }

  async verify(message: string, signature: string, publicKey: PublicKey): Promise<boolean> {
    const hash = crypto.createHash('sha256');
    hash.update(message);
    return this.verifySignature(hash.digest('hex'), signature, publicKey);
  }

  private signHash(hash: string, privateKey: PrivateKey): string {
    const signature = (parseInt(hash, 16) * privateKey.d) % privateKey.q;
    return signature.toString(16);
  }

  private verifySignature(hash: string, signature: string, publicKey: PublicKey): boolean {
    const expectedSignature = (parseInt(hash, 16) * publicKey.g) % publicKey.q;
    return expectedSignature.toString(16) === signature;
  }
}

export default QRCUtils;
