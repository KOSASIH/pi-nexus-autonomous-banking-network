pragma solidity ^0.8.0;

import "./PiNetwork.sol";
import "./PiNetworkRegistry.sol";

contract PiNetworkBridge {
    // Mapping of Pi Network contracts to their bridges
    mapping (address => address) public piNetworkBridges;

    // Event emitted when a Pi Network contract is bridged
    event PiNetworkBridgedEvent(address indexed piNetwork, address indexed bridge);

    // Function to bridge a Pi Network contract
    function bridgePiNetwork(PiNetwork piNetwork) public {
        piNetworkBridges[address(piNetwork)] = msg.sender;
        emit PiNetworkBridgedEvent(address(piNetwork), msg.sender);
    }

    // Function to get a Pi Network contract by bridge
    function getPiNetworkByBridge(address bridge) public view returns (PiNetwork) {
        return PiNetwork(piNetworkBridges[bridge]);
    }
}
