pragma solidity ^0.8.0;

import "./SidraBridge.sol";
import "./BridgeRegistry.sol";

contract SidraChainManager {
    SidraBridge public sidraBridge;
    BridgeRegistry public bridgeRegistry;

    event BridgeAdded(address indexed bridgeAddress);
    event BridgeRemoved(address indexed bridgeAddress);

    constructor() public {
        sidraBridge = SidraBridge(address(new SidraBridge()));
        bridgeRegistry = BridgeRegistry(address(new BridgeRegistry()));
    }

    function addBridge(address _bridgeAddress) public {
        require(!bridgeRegistry.isBridge(_bridgeAddress), "Bridge already added");
        bridgeRegistry.registerBridge(_bridgeAddress);
        sidraBridge.setForeignChainAddress(_bridgeAddress);
        emit BridgeAdded(_bridgeAddress);
    }

    function removeBridge(address _bridgeAddress) public {
        require(bridgeRegistry.isBridge(_bridgeAddress), "Bridge not added");
        bridgeRegistry.unregisterBridge(_bridgeAddress);
        sidraBridge.setForeignChainAddress(address(0));
        emit BridgeRemoved(_bridgeAddress);
    }

    function getBridgeAddress(address _bridgeAddress) public view returns (address) {
        return bridgeRegistry.getBridgeAddress(_bridgeAddress);
    }
}
