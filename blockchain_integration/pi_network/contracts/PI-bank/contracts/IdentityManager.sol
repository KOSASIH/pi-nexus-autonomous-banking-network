pragma solidity ^0.8.0;

contract IdentityManager {
    mapping (address => bytes32) public identities;

    function createIdentity(bytes32 identity) public {
        identities[msg.sender] = identity;
    }

    function updateIdentity(bytes32 newIdentity) public {
        require(identities[msg.sender]!= 0, "Identity not found");
        identities[msg.sender] = newIdentity;
    }

    function getIdentity(address user) public view returns (bytes32) {
        return identities[user];
    }
}
