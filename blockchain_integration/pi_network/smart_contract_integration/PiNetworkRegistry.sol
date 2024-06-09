pragma solidity ^0.8.0;

import "./PiNetworkFactory.sol";

contract PiNetworkRegistry {
    // Mapping of Pi Network contracts to their creators
    mapping (address => address) public piNetworkCreators;

    // Event emitted when a Pi Network contract is registered
    event PiNetworkRegisteredEvent(address indexed piNetwork, address indexed creator);

    // Function to register a Pi Network contract
    function registerPiNetwork(PiNetwork piNetwork) public {
        piNetworkCreators[address(piNetwork)] = msg.sender;
        emit PiNetworkRegisteredEvent(address(piNetwork), msg.sender);
    }

    // Function to get a Pi Network contract by address
    function getPiNetworkByAddress(address piNetwork) public view returns (PiNetwork) {
        return PiNetwork(piNetwork);
    }
}
