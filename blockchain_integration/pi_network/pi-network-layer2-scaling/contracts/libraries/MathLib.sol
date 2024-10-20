// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library MathLib {
    // Safe addition
    function safeAdd(uint256 a, uint256 b) internal pure returns (uint256) {
        require(a + b >= a, "MathLib: addition overflow");
        return a + b;
    }

    // Safe subtraction
    function safeSub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "MathLib: subtraction underflow");
        return a - b;
    }

    // Safe multiplication
    function safeMul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        require(a * b / a == b, "MathLib: multiplication overflow");
        return a * b;
    }

    // Safe division
    function safeDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "MathLib: division by zero");
        return a / b;
    }

    // Calculate the average of two numbers
    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        return safeDiv(safeAdd(a, b), 2);
    }

    // Calculate the maximum of two numbers
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a : b;
    }

    // Calculate the minimum of two numbers
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}
