import * as crypto from 'crypto';

class Crypto {
  static sha256(data) {
    const hash = crypto.createHash('sha256');
    hash.update(data);
    return hash.digest('hex');
  }
}

export { Crypto };
