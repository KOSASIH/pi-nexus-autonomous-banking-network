pragma solidity ^0.8.0;

library Strings {
    function toString(uint256 n) internal pure returns (string memory) {
        if (n == 0) {
            return "0";
        }
        uint256 len = 0;
        uint256 m = n;
        while (m > 0) {
            len++;
            m /= 10;
        }
        bytes memory bstr = new bytes(len);
        uint256 i = len - 1;
        while (n > 0) {
            uint256 rem = n % 10;
            bstr[i--] = bytes1(48 + rem);
            n /= 10;
        }
        return string(bstr);
    }

    function concat(string memory a, string memory b) internal pure returns (string memory) {
        uint256 aLen = bytes(a).length;
        uint256 bLen = bytes(b).length;
        bytes memory ab = new bytes(aLen + bLen);
        uint256 i;
        for (i = 0; i < aLen; i++) {
            ab[i] = a[i];
        }
        for (i = 0; i < bLen; i++) {
            ab[aLen + i] = b[i];
        }
        return string(ab);
    }

    function toHexString(uint256 n) internal pure returns (string memory) {
        if (n == 0) {
            return "0x0";
        }
        uint256 len = 0;
        uint256 m = n;
        while (m > 0) {
            len++;
            m >>= 4;
        }
        bytes memory bstr = new bytes(2 * len + 2);
        bstr[0] = '0';
        bstr[1] = 'x';
        uint256 i = 2 * len;
        while (n > 0) {
            uint256 rem = n & 0xf;
            bstr[i--] = bytes1(48 + rem);
            n >>= 4;
        }
        return string(bstr);
    }
}
