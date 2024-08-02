import { createHash } from 'crypto';
import { elliptic } from 'elliptic';

const hashAlgorithms = {
  SHA256: 'ha256',
  SHA512: 'ha512',
  KECCAK256: 'keccak256',
  RIPEMD160: 'ripemd160',
};

const ellipticCurves = {
  SECP256K1: 'ecp256k1',
  ED25519: 'ed25519',
};

class Hash {
  static hash(data, algorithm = hashAlgorithms.SHA256) {
    const hash = createHash(algorithm);
    hash.update(data);
    return hash.digest('hex');
  }

  static ellipticHash(data, curve = ellipticCurves.SECP256K1) {
    const ec = elliptic(curve);
    return ec.hash(data);
  }

  static scryptHash(data, options = { N: 16384, r: 8, p: 1 }) {
    return scrypt.hash(data, options);
  }

  static keccak256(data) {
    return this.hash(data, hashAlgorithms.KECCAK256);
  }

  static ripemd160(data) {
    return this.hash(data, hashAlgorithms.RIPEMD160);
  }
}

export default Hash;
