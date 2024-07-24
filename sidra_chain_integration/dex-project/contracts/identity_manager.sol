pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Strings.sol";

contract IdentityManager is Ownable {
    // Mapping of user addresses to their respective identities
    mapping(address => Identity) public identities;

    // Event emitted when a new identity is created
    event NewIdentity(address indexed user, string name, string email);

    // Event emitted when an identity is updated
    event UpdateIdentity(address indexed user, string name, string email);

    // Event emitted when an identity is verified
    event VerifyIdentity(address indexed user, bool verified);

    // Function to create a new identity
    function createIdentity(string memory name, string memory email) public {
        // Check if the user already has an identity
        require(identities[msg.sender] == Identity(0), "User already has an identity");

        // Create a new identity
        Identity memory identity = Identity(name, email, false);

        // Add the identity to the mapping
        identities[msg.sender] = identity;

        // Emit the NewIdentity event
        emit NewIdentity(msg.sender, name, email);
    }

    // Function to update an identity
    function updateIdentity(string memory name, string memory email) public {
        // Check if the user has an identity
        require(identities[msg.sender]!= Identity(0), "User does not have an identity");

        // Update the identity
        identities[msg.sender].name = name;
        identities[msg.sender].email = email;

        // Emit the UpdateIdentity event
        emit UpdateIdentity(msg.sender, name, email);
    }

    // Function to verify an identity
    function verifyIdentity(address user, bool verified) public onlyOwner {
        // Check if the user has an identity
        require(identities[user]!= Identity(0), "User does not have an identity");

        // Update the identity verification status
        identities[user].verified = verified;

        // Emit the VerifyIdentity event
        emit VerifyIdentity(user, verified);
    }

    // Function to get an identity
    function getIdentity(address user) public view returns (Identity memory) {
        return identities[user];
    }

    // Identity struct
    struct Identity {
        string name;
        string email;
        bool verified;
    }
}
