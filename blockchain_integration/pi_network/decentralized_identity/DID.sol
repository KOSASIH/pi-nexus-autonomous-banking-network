// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DecentralizedIdentity {
    struct Identity {
        string name;
        string email;
        bool exists;
    }

    mapping(address => Identity) private identities;

    event IdentityRegistered(address indexed user, string name, string email);
    event IdentityUpdated(address indexed user, string name, string email);

    // Register a new identity
    function registerIdentity(string memory _name, string memory _email) public {
        require(!identities[msg.sender].exists, "Identity already exists.");
        identities[msg.sender] = Identity(_name, _email, true);
        emit IdentityRegistered(msg.sender, _name, _email);
    }

    // Update existing identity
    function updateIdentity(string memory _name, string memory _email) public {
        require(identities[msg.sender].exists, "Identity does not exist.");
        identities[msg.sender].name = _name;
        identities[msg.sender].email = _email;
        emit IdentityUpdated(msg.sender, _name, _email);
    }

    // Retrieve identity information
    function getIdentity(address _user) public view returns (string memory, string memory) {
        require(identities[_user].exists, "Identity does not exist.");
        return (identities[_user].name, identities[_user].email);
    }

    // Check if identity exists
    function identityExists(address _user) public view returns (bool) {
        return identities[_user].exists;
    }
}
