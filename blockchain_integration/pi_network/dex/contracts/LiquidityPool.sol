pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/SafeERC20.sol";

contract LiquidityPool {
    mapping(address => uint256) public liquidityProviders;
    mapping(address => uint256) public liquidityTokens;

    function addLiquidity(IERC20 tokenA, IERC20 tokenB, uint256 amountA, uint256 amountB) external {
        //...
    }

    function removeLiquidity(IERC20 tokenA, IERC20 tokenB, uint256 liquidityTokens) external {
        //...
    }

    function getLiquidityProvider(address user) public view returns (uint256) {
        //...
    }

    function getLiquidityToken(address user) public view returns (uint256) {
        //...
    }
}
