pragma solidity ^0.8.0;

import "https://github.com/stellar/solidity-stellar/blob/master/contracts/StellarToken.sol";

contract IdentityManager {
    address private owner;
    mapping (address => bytes32) public identities;

    constructor() public {
        owner = msg.sender;
    }

    function createIdentity(bytes32 _identity) public {
        require(msg.sender == owner, "Only the owner can create identities");
        identities[msg.sender] = _identity;
    }

    function getIdentity(address _address) public view returns (bytes32) {
        return identities[_address];
    }

    function updateIdentity(bytes32 _newIdentity) public {
        require(msg.sender == owner, "Only the owner can update identities");
        identities[msg.sender] = _newIdentity;
    }
}
