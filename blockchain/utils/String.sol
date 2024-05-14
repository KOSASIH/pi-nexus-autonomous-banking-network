// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

library String {
    function concat(string memory a, string memory b) internal pure returns (string memory) {
        return string(abi.encodePacked(a, b));
    }

    function length(string memory s) internal view returns (uint256) {
        return s.length;
    }

    function substring(string memory s, uint256 start, uint256 end) internal viewreturns (string memory) {
        require(start < end, "String: substring start > end");
        require(end <= s.length, "String: substring end > length");

        bytes memory bytesS = bytes(s);
        bytes memory bytesSubstring = new bytes(end - start);
        for (uint256 i = start; i < end; i++) {
            bytesSubstring[i - start] = bytesS[i];
        }

        return string(bytesSubstring);
    }

    function trim(string memory s) internal view returns (string memory) {
        uint256 start = 0;
        uint256 end = s.length;

        while (start < end && s[start] == ' ') {
            start++;
        }

        while (end > start && s[end - 1] == ' ') {
            end--;
        }

        return substring(s, start, end);
    }

    function toUpper(string memory s) internal view returns (string memory) {
        bytes memory bytesS = bytes(s);
        for (uint256 i = 0; i < bytesS.length; i++) {
            bytesS[i] = bytes1(uint8(bytesS[i]) & 0xDF);
        }
        return string(bytesS);
    }

    function toLowerCase(string memory s) internal view returns (string memory) {
        bytes memory bytesS = bytes(s);
        for (uint256 i = 0; i < bytesS.length; i++) {
            bytesS[i] = bytes1(uint8(bytesS[i]) | 0x20);
        }
        return string(bytesS);
    }
}
