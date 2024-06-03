pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract Uniswap {
    // Mapping of token reserves
    mapping (address => uint256) public reserves;

    // Mapping of token prices
    mapping (address => uint256) public prices;

    // Event emitted when a user adds liquidity
    event AddLiquidity(address indexed user, uint256 amount);

    // Event emitted when a user removes liquidity
    event RemoveLiquidity(address indexed user, uint256 amount);

    // Event emitted when a user swaps tokens
    event Swap(address indexed user, uint256 amountIn, uint256 amountOut);

    // Function to add liquidity
    function addLiquidity(uint256 amount) public {
        require(amount > 0, "Invalid liquidity amount");
        reserves[msg.sender] += amount;
        emit AddLiquidity(msg.sender, amount);
    }

    // Function to remove liquidity
    function removeLiquidity(uint256 amount) public {
        require(amount > 0, "Invalid liquidity amount");
        require(reserves[msg.sender] >= amount, "Insufficient liquidity");
        reserves[msg.sender] -= amount;
        emit RemoveLiquidity(msg.sender, amount);
    }

    // Function to swap tokens
    function swap(uint256 amountIn, address tokenIn, address tokenOut) public {
        require(amountIn > 0, "Invalid swap amount");
        require(prices[tokenIn] > 0, "Token price not set");
        require(prices[tokenOut] > 0, "Token price not set");
        uint256 amountOut = amountIn * prices[tokenOut] / prices[tokenIn];
        require(reserves[tokenOut] >= amountOut, "Insufficient liquidity");
        reserves[tokenOut] -= amountOut;
        emit Swap(msg.sender, amountIn, amountOut);
    }

    // Function to set token prices
    functionsetPrice(address token, uint256 price) public {
        prices[token] = price;
    }
}
