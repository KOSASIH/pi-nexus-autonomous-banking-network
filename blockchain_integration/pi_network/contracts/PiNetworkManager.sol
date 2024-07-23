pragma solidity ^0.8.0;

import "./PiNetwork.sol";

contract PiNetworkManager {
    address private owner;
    PiNetwork private piNetwork;

    constructor() {
        owner = msg.sender;
    }

    function setPiNetwork(address _piNetwork) public {
        piNetwork = PiNetwork(_piNetwork);
    }

    function startPiNetwork() public {
        piNetwork.startStellarQuickstart();
    }

    function createPiConsensusContainer() public {
        piNetwork.createPiConsensusContainer();
    }
}
