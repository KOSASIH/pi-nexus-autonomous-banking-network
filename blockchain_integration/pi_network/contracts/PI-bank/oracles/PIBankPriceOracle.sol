pragma solidity ^0.8.0;

import "./IPIBankPriceOracle.sol";

contract PIBankPriceOracle is IPIBankPriceOracle {
    mapping(string => uint256) public prices;

    function fetchPrice(string calldata _symbol) public {
        // implement price fetching logic
    }

    function updatePrice(string calldata _symbol, uint256 _price) public {
        prices[_symbol] = _price;
    }

    function getPrice(string calldata _symbol) public view returns (uint256) {
        return prices[_symbol];
    }
}
