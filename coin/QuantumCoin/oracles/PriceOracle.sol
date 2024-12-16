// contracts/oracles/PriceOracle.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract PriceOracle is Ownable {
    // Mapping to store prices for different assets
    mapping(string => uint256) private prices;

    // Event to emit when a price is updated
    event PriceUpdated(string asset, uint256 price);

    // Function to update the price of an asset
    function updatePrice(string memory asset, uint256 price) external onlyOwner {
        prices[asset] = price;
        emit PriceUpdated(asset, price);
    }

    // Function to get the price of an asset
    function getPrice(string memory asset) external view returns (uint256) {
        return prices[asset];
    }
}
