pragma solidity ^0.8.0;

library KosasihUniversalisMath {
    /**
     * @dev Returns the result of adding two unsigned integers, reverting on overflow.
     */
    function safeAdd(uint256 _a, uint256 _b) internal pure returns (uint256) {
        uint256 c = _a + _b;
        require(c >= _a, "Addition overflow");
        return c;
    }

    /**
     * @dev Returns the result of subtracting one unsigned integer from another, reverting on underflow.
     */
    function safeSub(uint256 _a, uint256 _b) internal pure returns (uint256) {
        require(_b <= _a, "Subtraction underflow");
        return _a - _b;
    }

    /**
     * @dev Returns the result of multiplying two unsigned integers, reverting on overflow.
     */
    function safeMul(uint256 _a, uint256 _b) internal pure returns (uint256) {
        if (_a == 0) {
            return 0;
        }
        uint256 c = _a * _b;
        require(c / _a == _b, "Multiplication overflow");
        return c;
    }

    /**
     * @dev Returns the result of dividing one unsigned integer by another, reverting on division by zero.
     */
    function safeDiv(uint256 _a, uint256 _b) internal pure returns (uint256) {
        require(_b > 0, "Division by zero");
        return _a / _b;
    }

    /**
     * @dev Returns the remainder of dividing one unsigned integer by another, reverting on division by zero.
     */
    function safeMod(uint256 _a, uint256 _b) internal pure returns (uint256) {
        require(_b > 0, "Division by zero");
        return _a % _b;
    }

    /**
     * @dev Returns the square root of a given unsigned integer.
     */
    function sqrt(uint256 _x) internal pure returns (uint256) {
        uint256 z = (_x + 1) / 2;
        uint256 y = _x;
        while (z < y) {
            y = z;
            z = (_x / z + z) / 2;
        }
        return y;
    }

    /**
     * @dev Returns the absolute value of a given signed integer.
     */
    function abs(int256 _x) internal pure returns (uint256) {
        return _x >= 0? uint256(_x) : uint256(-_x);
    }
}
