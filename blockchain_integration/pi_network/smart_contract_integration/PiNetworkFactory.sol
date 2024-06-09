pragma solidity ^0.8.0;

import "./PiNetwork.sol";

contract PiNetworkFactory {
    // Mapping of Pi Network contracts
    mapping (address => PiNetwork) public piNetworks;

    // Event emitted when a new Pi Network contract is created
    event NewPiNetworkEvent(address indexed creator, address piNetwork);

    // Function to create a new Pi Network contract
    function createPiNetwork() public {
        PiNetwork piNetwork = new PiNetwork();
        piNetworks[msg.sender] = piNetwork;
        emit NewPiNetworkEvent(msg.sender, address(piNetwork));
    }

    // Function to get a Pi Network contract by creator
    function getPiNetworkByCreator(address creator) public view returns (PiNetwork) {
        return piNetworks[creator];
    }
}
