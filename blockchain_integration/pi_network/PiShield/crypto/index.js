const { LatticeCrypto } = require('./lattice_crypto');
const { HashBasedSignatures } = require('./hash_based_signatures');

class Crypto {
    async encrypt(data) {
        const latticeCrypto = new LatticeCrypto();
        return latticeCrypto.encrypt(data);
    }

    async decrypt(encryptedData) {
        const latticeCrypto = new LatticeCrypto();
        return latticeCrypto.decrypt(encryptedData);
    }

    async sign(data) {
        const hashBasedSignatures = new HashBasedSignatures();
        return hashBasedSignatures.sign(data);
    }

    async verify(signature, data) {
        const hashBasedSignatures = new HashBasedSignatures();
        return hashBasedSignatures.verify(signature, data);
    }
}

module.exports = Crypto;
