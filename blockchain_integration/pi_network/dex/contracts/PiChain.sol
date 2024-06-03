pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract PiChain {
    using SafeMath for uint256;

    struct Block {
        uint256 timestamp;
        uint256 nonce;
        bytes32 hash;
        bytes32 previousHash;
        uint256 transactionsRoot;
    }

    Block[] public blockchain;

    event BlockAdded(uint256 blockNumber, uint256 timestamp, bytes32 hash);

    function addBlock(uint256 nonce, bytes32 hash, bytes32 previousHash, uint256 transactionsRoot) public {
        Block memory block = Block(block.timestamp, nonce, hash, previousHash, transactionsRoot);
        blockchain.push(block);
        emit BlockAdded(blockchain.length - 1, block.timestamp, hash);
    }

    function getBlockByNumber(uint256 blockNumber) public view returns (Block memory) {
        return blockchain[blockNumber];
    }

    function getBlockByHash(bytes32 hash) public view returns (Block memory) {
        for (uint256 i = 0; i < blockchain.length; i++) {
            if (blockchain[i].hash == hash) {
                return blockchain[i];
            }
        }
        revert("Block not found");
    }

    function getBlockchainLength() public view returns (uint256) {
        return blockchain.length;
    }
}
