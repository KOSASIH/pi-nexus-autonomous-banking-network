// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract AMMDEX is Ownable {
    using SafeERC20 for IERC20;

    struct TokenPair {
        IERC20 tokenA;
        IERC20 tokenB;
        uint256 reserveA;
        uint256 reserveB;
    }

    mapping(address => TokenPair) public tokenPairs;

    event TokenPairCreated(address indexed pairAddress, address indexed tokenA, address indexed tokenB);
    event Swap(address indexed pairAddress, address indexed tokenIn, address indexed tokenOut, uint256 amountIn, uint256 amountOut);

    function createTokenPair(IERC20 tokenA, IERC20 tokenB) external onlyOwner {
        require(tokenA!= tokenB, "Tokens must be different");
        require(tokenPairs[address(tokenA)].tokenA == IERC20(0), "Token A already exists in a pair");
        require(tokenPairs[address(tokenB)].tokenB == IERC20(0), "Token B already exists in a pair");

        TokenPair storage pair = tokenPairs[address(this)];
        pair.tokenA = tokenA;
        pair.tokenB = tokenB;
        pair.reserveA = 0;
        pair.reserveB = 0;

        emit TokenPairCreated(address(this), address(tokenA), address(tokenB));
    }

    function addLiquidity(IERC20 tokenA, IERC20 tokenB, uint256 amountA, uint256 amountB) external {
        TokenPair storage pair = tokenPairs[address(this)];
        require(pair.tokenA == tokenA && pair.tokenB == tokenB, "Invalid token pair");

        tokenA.safeTransferFrom(msg.sender, address(this), amountA);
        tokenB.safeTransferFrom(msg.sender, address(this), amountB);

        pair.reserveA += amountA;
        pair.reserveB += amountB;
    }

    function removeLiquidity(IERC20 tokenA, IERC20 tokenB, uint256 amountA, uint256 amountB) external {
        TokenPair storage pair = tokenPairs[address(this)];
        require(pair.tokenA == tokenA && pair.tokenB == tokenB, "Invalid token pair");

        require(pair.reserveA >= amountA && pair.reserveB >= amountB, "Insufficient liquidity");

        tokenA.safeTransfer(msg.sender, amountA);
        tokenB.safeTransfer(msg.sender, amountB);

        pair.reserveA -= amountA;
        pair.reserveB -= amountB;
    }

    function swap(IERC20 tokenIn, IERC20 tokenOut, uint256 amountIn) external {
        TokenPair storage pair = tokenPairs[address(this)];
        require(pair.tokenA == tokenIn || pair.tokenB == tokenIn, "Invalid token in");

        uint256 amountOut;
        if (pair.tokenA == tokenIn) {
            amountOut = getAmountOut(amountIn, pair.reserveA, pair.reserveB);
            require(tokenOut.transfer(msg.sender, amountOut), "Transfer failed");
            pair.reserveA += amountIn;
            pair.reserveB -= amountOut;
        } else {
            amountOut = getAmountOut(amountIn, pair.reserveB, pair.reserveA);
            require(tokenOut.transfer(msg.sender, amountOut), "Transfer failed");
            pair.reserveB += amountIn;
            pair.reserveA -= amountOut;
        }

        emit Swap(address(this), address(tokenIn), address(tokenOut), amountIn, amountOut);
    }

    function getAmountOut(uint256 amountIn, uint256 reserveIn, uint256 reserveOut) public pure returns (uint256) {
        require(amountIn > 0, "Amount in must be greater than zero");
        require(reserveIn > 0 && reserveOut > 0, "Reserves must be greater than zero");

        uint256 amountOut = (amountIn * reserveOut) / (reserveIn + amountIn);
        return amountOut;
    }
}
