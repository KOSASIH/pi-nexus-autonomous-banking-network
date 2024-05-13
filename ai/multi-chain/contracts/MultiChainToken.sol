pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MultiChainToken is ERC20, Ownable {
    IMultiChainExchange public exchange;

    constructor(IMultiChainExchange exchange) {
        require(exchange != address(0), "Invalid exchange address");
        this.exchange = exchange;
    }

    function swapTokens(address tokenIn, uint256 amountIn, address tokenOut, uint256 amountOutMin) external {
        require(tokenIn != address(0), "Invalid token in address");
        require(tokenOut != address(0), "Invalid token out address");
        require(amountIn > 0, "Invalid token in amount");
        require(amountOutMin > 0, "Invalid token out minimum amount");
        require(balanceOf(msg.sender) >= amountIn, "Insufficient balance");
        require(exchange.getTokenBalance(tokenIn, address(this)) >= amountIn, "Insufficient token in balance");
        require(exchange.getTokenBalance(tokenOut, address(this)) >= amountOutMin, "Insufficient token out balance");
        _transfer(msg.sender, address(exchange), amountIn);
        exchange.swapTokens(tokenIn, amountIn, tokenOut, amountOutMin);
    }

    function getTokenBalance(address token, address account) external view returns (uint256) {
        return exchange.getTokenBalance(token, account);
    }

    function getExchangeRate(address tokenIn, address tokenOut) external view returns (uint256) {
        return exchange.getExchangeRate(tokenIn, tokenOut);
    }
}
