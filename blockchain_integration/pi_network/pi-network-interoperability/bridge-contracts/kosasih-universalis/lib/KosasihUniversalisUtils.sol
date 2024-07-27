pragma solidity ^0.8.0;

library KosasihUniversalisUtils {
    function bytesToAddress(bytes _bytes) internal pure returns (address) {
        // Convert bytes to address
        return address(uint160(uint256(_bytes)));
    }

    function addressToBytes(address _address) internal pure returns (bytes) {
        // Convert address to bytes
        return abi.encodePacked(_address);
    }
}
