// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IFlashLoanReceiver {
    function executeOperation(uint256 amount, uint256 fee) external;
}

contract FlashLoan {
    function flashLoan(address receiver, uint256 amount) external {
        uint256 fee = calculateFee(amount);
        // Logic to transfer the amount to the receiver
        IFlashLoanReceiver(receiver).executeOperation(amount, fee);
        // Logic to ensure the amount + fee is returned
    }

    function calculateFee(uint256 amount) internal pure returns (uint256) {
        return (amount * 1) / 100; // 1% fee
    }
}
