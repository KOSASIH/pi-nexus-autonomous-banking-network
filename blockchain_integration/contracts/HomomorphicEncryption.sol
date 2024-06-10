pragma solidity ^0.8.0;

import "https://github.com/homomorphic-encryption/homomorphic-encryption-solidity/contracts/HomomorphicEncryption.sol";

contract HomomorphicEncryption {
    HomomorphicEncryption public he;

    constructor() {
        he = new HomomorphicEncryption();
    }

    // Fully homomorphic encryption and decryption
    function encrypt(uint256[] memory plaintext) public {
        //...
    }

    function decrypt(uint256[] memory ciphertext) public {
        //...
    }
}
