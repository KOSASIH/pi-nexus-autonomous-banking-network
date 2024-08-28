pragma solidity ^0.8.0;

import "./ExchangeManager.sol";
import "./LiquidityProvider.sol";

contract PiCoinListing {
    using ExchangeManager for address;
    using LiquidityProvider for address;

    // Mapping of exchanges
    mapping (address => Exchange) public exchanges;

    // Struct to represent an exchange
    struct Exchange {
        address exchangeAddress;
        string exchangeName;
        bool listed;
    }

    // Event emitted when Pi Coin is listed on an exchange
    event PiCoinListed(address indexed exchangeAddress, string exchangeName);

    // Function to list Pi Coin on an exchange
    function listPiCoinOnExchange(address _exchangeAddress, string memory _exchangeName) public {
        Exchange storage exchange = exchanges[_exchangeAddress];
        exchange.exchangeAddress = _exchangeAddress;
        exchange.exchangeName = _exchangeName;
        exchange.listed = true;
        emit PiCoinListed(_exchangeAddress, _exchangeName);
    }

    // Function to get the list of exchanges where Pi Coin is listed
    function getExchanges() public view returns (Exchange[] memory) {
        Exchange[] memory exchangeList = new Exchange[](exchanges.length);
        for (address exchangeAddress in exchanges) {
            exchangeList.push(exchanges[exchangeAddress]);
        }
        return exchangeList;
    }
}
