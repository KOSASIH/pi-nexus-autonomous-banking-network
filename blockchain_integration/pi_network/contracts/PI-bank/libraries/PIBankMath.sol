pragma solidity ^0.8.0;

library PIBankMath {
    // Safe math operations
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "Addition overflow");
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "Subtraction underflow");
        return a - b;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "Multiplication overflow");
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "Division by zero");
        return a / b;
    }

    // Advanced math operations
    function pow(uint256 a, uint256 b) internal pure returns (uint256) {
        if (b == 0) {
            return 1;
        }
        uint256 c = a ** b;
        require(c >= a, "Exponentiation overflow");
        return c;
    }

    function sqrt(uint256 a) internal pure returns (uint256) {
        require(a >= 0, "Square root of negative number");
        uint256 c = sqrtInternal(a);
        return c;
    }

    function sqrtInternal(uint256 a) internal pure returns (uint256) {
        uint256 x = a / 2;
        uint256 y = (x + a / x) / 2;
        while (y < x) {
            x = y;
            y = (x + a / x) / 2;
        }
        return x;
    }
}
