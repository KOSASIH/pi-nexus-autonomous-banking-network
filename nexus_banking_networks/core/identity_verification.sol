pragma solidity ^0.8.0;

contract IdentityVerification {
    mapping (address => string) public identities;

    function registerIdentity(string memory _identity) public {
        identities[msg.sender] = _identity;
    }

    function verifyIdentity(address _address) public view returns (bool) {
        return keccak256(abi.encodePacked(identities[_address])) == keccak256(abi.encodePacked("verified"));
    }
}
