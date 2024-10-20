pragma solidity ^0.8.0;

contract IdentityVerifier {
    // Define the mapping of addresses to their identities
    mapping (address => Identity) public identities;

    // Define the struct for an identity
    struct Identity {
        string name;
        string email;
        string phoneNumber;
    }

    // Event emitted when an identity is created
    event IdentityCreated(address indexed identityOwner);

    // Event emitted when an identity is updated
    event IdentityUpdated(address indexed identityOwner);

    // Function to create a new identity
    function createIdentity(string memory _name, string memory _email, string memory _phoneNumber) public {
        // Check if the identity already exists
        require(identities[msg.sender].name == "", "Identity already exists");

        // Create the identity
        identities[msg.sender].name = _name;
        identities[msg.sender].email = _email;
        identities[msg.sender].phoneNumber = _phoneNumber;

        // Emit the identity created event
        emit IdentityCreated(msg.sender);
    }

    // Function to update an identity
    function updateIdentity(string memory _name, string memory _email, string memory _phoneNumber) public {
        // Check if the identity exists
        require(identities[msg.sender].name != "", "Identity does not exist");

        // Update the identity
        identities[msg.sender].name = _name;
        identities[msg.sender].email = _email;
        identities[msg.sender].phoneNumber = _phoneNumber;

        // Emit the identity updated event
        emit IdentityUpdated(msg.sender);
    }

    // Function to get an identity
    function getIdentity(address _address) public view returns (Identity memory) {
        // Return the identity
        return identities[_address];
    }
}
