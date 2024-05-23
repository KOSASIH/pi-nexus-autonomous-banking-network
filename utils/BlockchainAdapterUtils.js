const Bitcoin = require('bitcoinjs-lib');
const Litecoin = require('litecoin-js');

class BlockchainAdapterUtils {

    // The function to convert a PI Network address to a Bitcoin address
    static convertPiToBitcoin(piAddress) {
        // Generate a Bitcoin address from the PI Network address
        const bitcoinAddress = Bitcoin.address.fromBase58Check(piAddress);

        // Return the Bitcoin address
        return bitcoinAddress;
    }

    // The function to convert a PI Network address to a Litecoin address
    static convertPiToLitecoin(piAddress) {
        // Generate a Litecoin address from the PI Network address
        const litecoinAddress = Litecoin.address.fromBase58Check(piAddress);

        // Return the Litecoin address
        return litecoinAddress;
    }

    // The function to convert a Bitcoin address to a PI Network address
    static convertBitcoinToPi(bitcoinAddress) {
        // Generate a PI Network address from the Bitcoin address
        const piAddress = Bitcoin.address.toBase58Check(bitcoinAddress);

        // Return the PI Network address
        return piAddress;
    }

    // The function to convert a Litecoin address to a PI Network address
    static convertLitecoinToPi(litecoinAddress) {
        // Generate a PI Network address from the Litecoin address
        const piAddress = Litecoin.address.toBase58Check(litecoinAddress);

        // Return the PI Network address
        return piAddress;
    }

}

module.exports = BlockchainAdapterUtils;
