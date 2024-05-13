pragma solidity ^0.8.0;

interface IMultiChainExchange {
    function swapTokens(address tokenIn, uint256 amountIn, address tokenOut, uint256 amountOutMin) external;
    function getTokenBalance(address token, address account) external view returns (uint256);
    function getExchangeRate(address tokenIn, address tokenOut) external view returns (uint256);
}
