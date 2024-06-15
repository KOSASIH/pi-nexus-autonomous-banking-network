pragma solidity ^0.8.0;

library PIBankUtils {
    // String utility functions
    function toString(uint256 value) internal pure returns (string memory) {
        return uint2str(value);
    }

    function uint2str(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits--;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }

    // Address utility functions
    function isContract(address addr) internal view returns (bool) {
        uint256 size;
        assembly {
            size := extcodesize(addr)
        }
        return size > 0;
    }

    function getContractName(address addr) internal view returns (string memory) {
        if (!isContract(addr)) {
            return "";
        }
        bytes32 name;
        assembly {
            name := extcodehash(addr)
        }
        return bytes32ToString(name);
    }

    function bytes32ToString(bytes32 x) internal pure returns (string memory) {
        bytes memory bytesString = new bytes(32);
        for (uint256 i = 0; i < 32; i++) {
            bytesString[i] = bytes1(uint8(uint256(x) / (2**(8*(31 - i)))));
        }
        return string(bytesString);
    }
}
