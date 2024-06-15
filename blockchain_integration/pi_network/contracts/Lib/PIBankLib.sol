pragma solidity ^0.8.0;

library PIBankLib {
    // Function to calculate interest
    function calculateInterest(uint256 balance, uint256 rate) internal pure returns (uint256) {
        return balance.mul(rate).div(100);
    }
}
