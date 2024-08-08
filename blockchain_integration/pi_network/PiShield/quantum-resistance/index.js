const { LatticeBasedCrypto } = require('./lattice_based_crypto');
const { HashBasedSignatures } = require('./hash_based_signatures');

class QuantumResistance {
    async encrypt(data) {
        const latticeBasedCrypto = new LatticeBasedCrypto();
        return latticeBasedCrypto.encrypt(data);
    }

    async decrypt(encryptedData) {
        const latticeBasedCrypto = new LatticeBasedCrypto();
        return latticeBasedCrypto.decrypt(encryptedData);
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

module.exports = QuantumResistance;
