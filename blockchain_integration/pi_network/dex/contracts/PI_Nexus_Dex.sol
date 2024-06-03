pragma solidity ^0.8.0;

import "https://github.com/Uniswap/uniswap-v2-core/blob/master/contracts/UniswapV2Pair.sol";

contract PI_Nexus_Dex {
    address private owner;
    mapping (address => mapping (address => uint256)) public liquidity;
    mapping (address => mapping (address => uint256)) public reserves;

    constructor() public {
        owner = msg.sender;
    }

    function addLiquidity(address tokenA, address tokenB, uint256 amountA, uint256 amountB) public {
        require(msg.sender == owner, "Only the owner can add liquidity");
        liquidity[tokenA][tokenB] += amountA;
        liquidity[tokenB][tokenA] += amountB;
        reserves[tokenA] += amountA;
        reserves[tokenB] += amountB;
    }

    function removeLiquidity(address tokenA, address tokenB, uint256 amountA, uint256 amountB) public {
        require(msg.sender == owner, "Only the owner can remove liquidity");
        liquidity[tokenA][tokenB] -= amountA;
        liquidity[tokenB][tokenA] -= amountB;
        reserves[tokenA] -= amountA;
        reserves[tokenB] -= amountB;
    }

    function getReserves(address tokenA, address tokenB) public view returns (uint256, uint256) {
        return (reserves[tokenA], reserves[tokenB]);
    }

    function getAmountOut(address tokenIn, address tokenOut, uint256 amountIn) public view returns (uint256) {
        (uint256 reserveIn, uint256 reserveOut) = getReserves(tokenIn, tokenOut);
        uint256 amountOut = (amountIn * reserveOut) / reserveIn;
        return amountOut;
    }

    function swap(address tokenIn, address tokenOut, uint256 amountIn) public {
        require(msg.sender == owner, "Only the owner can swap");
        uint256 amountOut = getAmountOut(tokenIn, tokenOut, amountIn);
        reserves[tokenIn] -= amountIn;
        reserves[tokenOut] += amountOut;
        emit Swap(tokenIn, tokenOut, amountIn, amountOut);
    }
}
