pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/SafeERC20.sol";

contract AutomatedMarketMaker {
    mapping(address => uint256) public reserves;

    function getAmountOut(uint256 amountIn, uint256 reserveIn, uint256 reserveOut) public pure returns (uint256) {
        //...
    }

    function swap(IERC20 tokenIn, IERC20 tokenOut, uint256 amountIn) external {
        //...
    }
}
