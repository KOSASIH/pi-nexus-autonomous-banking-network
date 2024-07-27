pragma solidity ^0.8.0;

contract BridgeRegistry {
    mapping(address => address) public bridgeAddresses;
    mapping(address => bool) public isBridgeRegistered;

    event BridgeRegistered(address indexed bridgeAddress);
    event BridgeUnregistered(address indexed bridgeAddress);

    function registerBridge(address _bridgeAddress) public {
        require(!isBridgeRegistered[_bridgeAddress], "Bridge already registered");
        bridgeAddresses[_bridgeAddress] = _bridgeAddress;
        isBridgeRegistered[_bridgeAddress] = true;
        emit BridgeRegistered(_bridgeAddress);
    }

    function unregisterBridge(address _bridgeAddress) public {
        require(isBridgeRegistered[_bridgeAddress], "Bridge not registered");
        delete bridgeAddresses[_bridgeAddress];
        isBridgeRegistered[_bridgeAddress] = false;
        emit BridgeUnregistered(_bridgeAddress);
    }

    function getBridgeAddress(address _bridgeAddress) public view returns (address) {
        return bridgeAddresses[_bridgeAddress];
    }

    function isBridge(address _bridgeAddress) public view returns (bool) {
        return isBridgeRegistered[_bridgeAddress];
    }
}
