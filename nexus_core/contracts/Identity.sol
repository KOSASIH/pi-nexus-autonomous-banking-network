pragma solidity ^0.8.0;

contract Identity {
    mapping (address => string) public identities;

    constructor() {
        // Initialize identity mapping
    }

    function setIdentity(string memory identity) public {
        identities[msg.sender] = identity;
    }

    function getIdentity(address account) public view returns (string memory) {
        return identities[account];
    }
}
