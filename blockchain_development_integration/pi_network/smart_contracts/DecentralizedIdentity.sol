// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DecentralizedIdentity {
    struct Identity {
        string name;
        string email;
        string phone;
        bool exists;
    }

    mapping(address => Identity) public identities;

    function createIdentity(string memory name, string memory email, string memory phone) external {
        require(!identities[msg.sender].exists, "Identity already exists");
        identities[msg.sender] = Identity(name, email, phone, true);
    }

    function updateIdentity(string memory name, string memory email, string memory phone) external {
        require(identities[msg.sender].exists, "Identity does not exist");
        identities[msg.sender].name = name;
        identities[msg.sender].email = email;
        identities[msg.sender].phone = phone;
    }

    function getIdentity(address user) external view returns (string memory, string memory, string memory) {
        require(identities[user].exists, "Identity does not exist");
        Identity memory identity = identities[user];
        return (identity.name, identity.email, identity.phone);
    }
}
