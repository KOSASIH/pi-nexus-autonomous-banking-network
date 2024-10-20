pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract DEXContract {
    using SafeERC20 for IERC20;

    // Struct to represent a trade
    struct Trade {
        address trader;
        address tokenIn;
        address tokenOut;
        uint amountIn;
        uint amountOut;
        uint timestamp;
    }

    // Mapping of token pairs to their exchange rates
    mapping(address => mapping(address => uint)) public exchangeRates;

    // Array to store trades
    Trade[] public trades;

    // Event emitted when a trade is executed
    event TradeExecuted(address indexed trader, address tokenIn, address tokenOut, uint amountIn, uint amountOut);

    // Function to set the exchange rate for a token pair
    function setExchangeRate(address _tokenIn, address _tokenOut, uint _rate) public {
        exchangeRates[_tokenIn][_tokenOut] = _rate;
    }

    // Function to execute a trade
    function trade(address _tokenIn, address _tokenOut, uint _amountIn) public {
        uint rate = exchangeRates[_tokenIn][_tokenOut];
        require(rate > 0, "Exchange rate not set");
        
        uint amountOut = _amountIn * rate / 1e18; // Assuming rate is in 18 decimals

        // Transfer tokens from the trader to the contract
        IERC20(_tokenIn).safeTransferFrom(msg.sender, address(this), _amountIn);
        // Transfer tokens from the contract to the trader
        IERC20(_tokenOut).safeTransfer(msg.sender, amountOut);

        // Record the trade
        trades.push(Trade(msg.sender, _tokenIn, _tokenOut, _amountIn, amountOut, block.timestamp));

        // Emit the TradeExecuted event
        emit TradeExecuted(msg.sender, _tokenIn, _tokenOut, _amountIn, amountOut);
    }

    // Function to get the details of a trade
    function getTrade(uint _index) public view returns (Trade memory) {
        return trades[_index];
    }

    // Function to get the number of trades
    function getTradeCount() public view returns (uint) {
        return trades.length;
    }
}
