// identity_verification.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";

contract IdentityVerification {
    using Roles for address;

    struct Identity {
        address owner;
        string name;
        string email;
        string phoneNumber;
    }

    mapping (address => Identity) public identities;

    event IdentityCreated(address indexed owner, string name, string email, string phoneNumber);
    event IdentityUpdated(address indexed owner, string name, string email, string phoneNumber);

    function createIdentity(string memory _name, string memory _email, string memory _phoneNumber) public {
        require(msg.sender!= address(0), "Invalid sender");
        Identity storage identity = identities[msg.sender];
        identity.owner = msg.sender;
        identity.name = _name;
        identity.email = _email;
        identity.phoneNumber = _phoneNumber;
        emit IdentityCreated(msg.sender, _name, _email, _phoneNumber);
    }

    function updateIdentity(string memory _name, string memory _email, string memory _phoneNumber) public {
        require(msg.sender!= address(0), "Invalid sender");
        Identity storage identity = identities[msg.sender];
        identity.name = _name;
        identity.email = _email;
        identity.phoneNumber = _phoneNumber;
        emit IdentityUpdated(msg.sender, _name, _email, _phoneNumber);
    }

    function getIdentity(address _owner) public view returns (string memory, string memory, string memory) {
        Identity storage identity = identities[_owner];
        return (identity.name, identity.email, identity.phoneNumber);
    }
}
