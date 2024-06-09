pragma solidity ^0.8.0;

import "./PiNetwork.sol";

contract PiNode {
    address public owner;
    PiNetwork public piNetwork;

    constructor() public {
        owner = msg.sender;
        piNetwork = PiNetwork(msg.sender);
    }

    function getAddress() public view returns (address) {
        return address(this);
    }

    function getNodeInfo() public view returns (string memory) {
        return "Pi Node: " + address(this);
    }
}
