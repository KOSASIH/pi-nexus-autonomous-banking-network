// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PriceCalculator {
    using SafeMath for uint256;

    // Mapping of token addresses to their corresponding price feeds
    mapping(address => address) public priceFeeds;

    // Function to set the price feed for a token
    function setPriceFeed(address token, address priceFeed) public {
        priceFeeds[token] = priceFeed;
    }

    // Function to get the price for a token
    function getPrice(address token) public view returns (uint256) {
        address priceFeed = priceFeeds[token];
        require(priceFeed!= address(0), "Price feed not set");

        // Get the price from the price feed
        uint256 price = IPriceFeed(priceFeed).getPrice();

        return price;
    }
}
