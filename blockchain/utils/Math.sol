// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

library Math {
    function safeAdd(uint256 a, uint256 b) internal pure returns (uint256) {
        require(a + b >= a, "Math: safeAdd underflow");
        return a + b;
    }

    function safeSub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "Math: safeSub underflow");
        return a - b;
    }

    function safeMul(uint256 a, uint256 b) internal pure returns (uint256) {
        require(a * b / a == b, "Math: safeMul overflow");
        return a * b;
    }

    function safeDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "Math: safeDiv by zero");
        return a / b;
    }

    function safeMod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "Math: safeMod by zero");
        return a % b;
    }

    function pow(uint256 base, uint256 exponent) internal pure returns (uint256) {
        if (exponent == 0) {
            return 1;
        }

        uint256 result = base;
        for (uint256 i = 1; i < exponent; i++) {
            result *= base;
        }

        return result;
    }

    function sqrt(uint256 x) internal pure returns (uint256) {
        if (x < 1) {
            return 0;
        }

        uint256 z = (uint256(1) << 256) / 2;
        uint256 y;
        while (z < y) {
            y = (z + x / z) / 2;
            z = y;
        }

        return z;
    }

    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }

    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}
