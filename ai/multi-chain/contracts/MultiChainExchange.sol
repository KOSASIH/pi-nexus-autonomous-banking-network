pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MultiChainExchange is Ownable {
    mapping(address => mapping(address => uint256)) public balances;

    function swapTokens(address tokenIn, uint256 amountIn, address tokenOut, uint256 amountOutMin) external {
        require(tokenIn != address(0), "Invalid token in address");
        require(tokenOut != address(0), "Invalid token out address");
        require(amountIn > 0, "Invalid token in amount");
        require(amountOutMin> 0, "Invalid token out minimum amount");
        require(balances[tokenIn][address(this)] >= amountIn, "Insufficient token in balance");
        require(balances[tokenOut][address(this)] >= amountOutMin, "Insufficient token out balance");
        IERC20(tokenIn).transferFrom(address(this), address(this), amountIn);
        IERC20(tokenOut).transfer(address(this), amountOutMin);
        balances[tokenIn][address(this)] -= amountIn;
        balances[tokenOut][address(this)] += amountOutMin;
    }

    function getTokenBalance(address token, address account) external view returns (uint256) {
        return IERC20(token).balanceOf(account);
    }

    function getExchangeRate(address tokenIn, address tokenOut) external view returns (uint256) {
        require(tokenIn != address(0), "Invalid token in address");
        require(tokenOut != address(0), "Invalid token out address");
        uint256 balanceIn = IERC20(tokenIn).balanceOf(address(this));
        uint256 balanceOut = IERC20(tokenOut).balanceOf(address(this));
        require(balanceIn > 0, "Insufficient token in balance");
        require(balanceOut > 0, "Insufficient token out balance");
        return balanceOut * 10**decimals() / balanceIn;
    }
}
