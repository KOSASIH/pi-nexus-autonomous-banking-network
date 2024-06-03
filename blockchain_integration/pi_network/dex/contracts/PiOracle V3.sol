pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/smartcontractkit/chainlink/blob/master/evm-contracts/src/v0.6/ChainlinkClient.sol";

contract PiOracleV3 {
    using SafeMath for uint256;

    struct Oracle {
        uint256 timestamp;
        uint256 price;
        uint256 decimals;
    }

    mapping(address => Oracle[]) public oracles;
    mapping(address => uint256) public latestPrice;

    event PriceUpdated(address indexed token, uint256 indexed price, uint256 indexed timestamp);

    function updatePrice(address token, uint256 price, uint256 decimals) public {
        Oracle[] storage tokenOracles = oracles[token];
        uint256 currentPrice = tokenOracles[tokenOracles.length - 1].price;

        if (tokenOracles.length == 0 || price!= currentPrice) {
            tokenOracles.push(Oracle(block.timestamp, price, decimals));
            latestPrice[token] = price;
            emit PriceUpdated(token, price, block.timestamp);
        }
    }

    function getOracleCount(address token) public view returns (uint256) {
        return oracles[token].length;
    }

    function getOraclePrice(address token, uint256 index) public view returns (uint256) {
        Oracle[] storage tokenOracles = oracles[token];
        require(index < tokenOracles.length, "Invalid oracle index");
        return tokenOracles[index].price;
    }

    function getOracleTimestamp(address token, uint256 index) public view returns (uint256) {
        Oracle[] storage tokenOracles = oracles[token];
        require(index < tokenOracles.length, "Invalid oracle index");
        return tokenOracles[index].timestamp;
    }

    function getOracleDecimals(address token, uint256 index) public view returns (uint256) {
        Oracle[] storage tokenOracles = oracles[token];
        require(index < tokenOracles.length, "Invalid oracle index");
        return tokenOracles[index].decimals;
    }
}
