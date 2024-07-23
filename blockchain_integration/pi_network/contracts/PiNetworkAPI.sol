pragma solidity ^0.8.0;

import "./PiNetwork.sol";

contract PiNetworkAPI {
    address private owner;
    PiNetwork private piNetwork;

    constructor() {
        owner = msg.sender;
    }

    function setPiNetwork(address _piNetwork) public {
        piNetwork = PiNetwork(_piNetwork);
    }

    function getApiUrl() public view returns (string memory) {
        return piNetwork.getApiUrl();
    }

    function setupTech() public {
        piNetwork.setupTech();
    }
}
