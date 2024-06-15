pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/security/ReentrancyGuard.sol";

contract DEXIntegration is ReentrancyGuard {
    using SafeMath for uint256;

    // Mapping of DEX pairs
    mapping (address => mapping (address => uint256)) public dexPairs;

    // Event emitted when a new DEX pair is created
    event NewDEXPair(address indexed tokenA, address indexed tokenB, uint256 indexed reserveA, uint256 indexed reserveB);

    // Event emitted when a DEX trade is executed
    event DEXTrade(address indexed tokenA, address indexed tokenB, uint256 indexed amountA, uint256 indexed amountB);

    // Function to create a new DEX pair
    function createDEXPair(address tokenA, address tokenB, uint256 reserveA, uint256 reserveB) public {
        require(!dexPairs[tokenA][tokenB].exists, "DEX pair already exists");
        dexPairs[tokenA][tokenB] = DEXPair(reserveA, reserveB, true);
        emit NewDEXPair(tokenA, tokenB, reserveA, reserveB);
    }

    // Function to execute a DEX trade
    function executeDEXTrade(address tokenA, address tokenB, uint256 amountA, uint256 amountB) public {
        require(dexPairs[tokenA][tokenB].exists, "DEX pair does not exist");
        require(dexPairs[tokenA][tokenB].reserveA >= amountA, "Insufficient reserveA");
        require(dexPairs[tokenA][tokenB].reserveB >= amountB, "Insufficient reserveB");
        dexPairs[tokenA][tokenB].reserveA = dexPairs[tokenA][tokenB].reserveA.sub(amountA);
        dexPairs[tokenA][tokenB].reserveB = dexPairs[tokenA][tokenB].reserveB.sub(amountB);
        emit DEXTrade(tokenA, tokenB, amountA, amountB);
    }
}

struct DEXPair {
    uint256 reserveA;
    uint256 reserveB;
    bool exists;
}
