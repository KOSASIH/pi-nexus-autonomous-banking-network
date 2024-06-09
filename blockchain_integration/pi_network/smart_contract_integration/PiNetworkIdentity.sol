pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract PiNetworkIdentity is Ownable {
    // Mapping of user addresses to their identities
    mapping (address => Identity) public identities;

    // Struct to represent a user identity
    struct Identity {
        string name;
        string email;
        string phoneNumber;
    }

    // Event emitted when a user's identity is updated
    event IdentityUpdateEvent(address indexed user, Identity identity);

    // Function to update a user's identity
    function updateIdentity(string memory name, string memory email, string memory phoneNumber) public {
        Identity storage identity = identities[msg.sender];
        identity.name = name;
        identity.email = email;
        identity.phoneNumber = phoneNumber;
        emit IdentityUpdateEvent(msg.sender, identity);
    }

    // Function to get a user's identity
    function getIdentity(address user) public view returns (Identity memory) {
        return identities[user];
    }
}
