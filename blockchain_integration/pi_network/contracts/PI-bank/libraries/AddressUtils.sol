pragma solidity ^0.8.0;

library AddressUtils {
    function isValidAddress(address _addr) internal pure returns (bool) {
        return _addr!= address(0);
    }

    function isContract(address _addr) internal view returns (bool) {
        uint256 size;
        assembly {
            size := extcodesize(_addr)
        }
        return size > 0;
    }

    function encodeAddress(address _addr) internal pure returns (bytes memory) {
        return abi.encodePacked(_addr);
    }
}
