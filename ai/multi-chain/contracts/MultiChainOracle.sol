pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract MultiChainOracle is Ownable {
    mapping(address => uint256) public prices;

    function setPrice(address token, uint256 price) external onlyOwner {
        require(price > 0, "Invalid price");
        prices[token] = price;
    }

    function getPrice(address token) external view returns (uint256) {
        return prices[token];
    }
}
