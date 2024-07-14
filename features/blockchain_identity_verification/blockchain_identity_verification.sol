// File name: blockchain_identity_verification.sol
pragma solidity ^0.8.0;

contract BlockchainIdentityVerification {
    address private owner;
    mapping (address => string) public identities;

    constructor() public {
        owner = msg.sender;
    }

    function verifyIdentity(address user, string identity) public {
        require(msg.sender == owner, "Only the owner can verify identities");
        identities[user] = identity;
    }

    function getIdentity(address user) public view returns (string) {
        return identities[user];
    }
}
