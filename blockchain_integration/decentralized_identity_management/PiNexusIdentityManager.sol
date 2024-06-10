// decentralized_identity_management/PiNexusIdentityManager.sol

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract PiNexusIdentityManager {
    // Mapping of user addresses to their respective identity information
    mapping (address => IdentityInfo) public identityInfo;

    // Event emitted when a new user is registered
    event NewUserRegistered(address indexed user, string name, string email);

    // Event emitted when a user's identity is updated
    event IdentityUpdated(address indexed user, string name, string email);

    // Struct to represent identity information
    struct IdentityInfo {
        string name;
        string email;
    }

    // Function to register a new user
    function registerUser(string memory _name, string memory _email) public {
        // Validate input data
        require(bytes(_name).length > 0, "Name cannot be empty");
        require(bytes(_email).length > 0, "Email cannot be empty");

        // Create a new identity info struct
        IdentityInfo memory newIdentityInfo = IdentityInfo(_name, _email);

        // Set the identity info for the user
        identityInfo[msg.sender] = newIdentityInfo;

        // Emit the NewUserRegistered event
        emit NewUserRegistered(msg.sender, _name, _email);
    }

    // Function to update a user's identity
    function updateIdentity(string memory _name, string memory _email) public {
        // Validate input data
        require(bytes(_name).length > 0, "Name cannot be empty");
        require(bytes(_email).length > 0, "Email cannot be empty");

        // Get the user's identity info
        IdentityInfo storage userIdentityInfo = identityInfo[msg.sender];

        // Update the user's identity info
        userIdentityInfo.name = _name;
        userIdentityInfo.email = _email;

        // Emit the IdentityUpdated event
        emit IdentityUpdated(msg.sender, _name, _email);
    }
}
