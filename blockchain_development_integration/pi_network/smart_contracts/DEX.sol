// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract DEX {
    function swapTokens(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOutMin,
        address to
    ) external {
        require(IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn), "Transfer failed");
        uint256 amountOut = getAmountOut(amountIn, tokenIn, tokenOut);
        require(amountOut >= amountOutMin, "Insufficient output amount");
        require(IERC20(tokenOut).transfer(to, amountOut), "Transfer failed");
    }

    function getAmountOut(uint256 amountIn, address tokenIn, address tokenOut) internal view returns (uint256) {
        // Implement your pricing logic here (e.g., using a constant product formula)
        return amountIn; // Placeholder logic
    }
}
