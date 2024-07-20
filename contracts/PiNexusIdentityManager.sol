pragma solidity ^0.6.0;

import "https://github.com/SidraChain/sidra-chain-contracts/blob/main/contracts/IdentityManager.sol";

contract PiNexusIdentityManager {
    address private owner;
    mapping (address => Identity) public identities;

    struct Identity {
        string name;
        string email;
        string phoneNumber;
        uint256 createdAt;
    }

    constructor() public {
        owner = msg.sender;
    }

    function createIdentity(string memory _name, string memory _email, string memory _phoneNumber) public {
        require(msg.sender == owner, "Only the owner can create identities");
        Identity memory newIdentity = Identity(_name, _email, _phoneNumber, block.timestamp);
        identities[msg.sender] = newIdentity;
    }

    function updateIdentity(string memory _name, string memory _email, string memory _phoneNumber) public {
        require(msg.sender == owner, "Only the owner can update identities");
        Identity storage identity = identities[msg.sender];
        identity.name = _name;
        identity.email = _email;
        identity.phoneNumber = _phoneNumber;
    }

    function verifyIdentity(address _address) public view returns (bool) {
        return identities[_address].createdAt > 0;
    }
}
