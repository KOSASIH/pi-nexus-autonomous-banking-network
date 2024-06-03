// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract FlashLoan {
    using SafeMath for uint256;
    using SafeERC20 for IERC20;

    // Event emitted when a flash loan is executed
    event FlashLoan(address user, address token, uint256 amount, uint256 fee);

    // Function to execute a flash loan
    function flashLoan(address token, uint256 amount) public {
        require(amount > 0, "Invalid amount");

        // Transfer the tokens to the user
        IERC20(token).safeTransferFrom(address(this), msg.sender, amount);

        // Calculate the fee
        uint256 fee = amount.mul(5).div(1000); // 0.5% fee

        // Transfer the fee back to the contract
        IERC20(token).safeTransferFrom(msg.sender, address(this), fee);

        emit FlashLoan(msg.sender, token, amount, fee);
    }
}
