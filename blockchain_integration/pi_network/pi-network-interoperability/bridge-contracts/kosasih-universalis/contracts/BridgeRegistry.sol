pragma solidity ^0.8.0;

contract BridgeRegistry {
    mapping (address => address) public bridges;

    function registerBridge(address _bridgeAddress) public {
        bridges[msg.sender] = _bridgeAddress;
    }

    function getBridge(address _address) public view returns (address) {
        return bridges[_address];
    }
}
