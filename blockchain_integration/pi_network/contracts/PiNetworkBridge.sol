// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract PiNetworkBridge {
    IERC20 private piToken;

    constructor(address piTokenAddress) {
        piToken = IERC20(piTokenAddress);
    }

    function deposit(uint256 amount) external {
        require(amount > 0, "Amount must be greater than zero");
        piToken.transferFrom(msg.sender, address(this), amount);
    }

    function withdraw(uint256 amount) external {
        require(amount > 0, "Amount must be greater than zero");
        require(piToken.balanceOf(address(this)) >= amount, "Insufficient balance");
        piToken.transfer(msg.sender, amount);
    }
}
