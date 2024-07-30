pragma solidity ^0.8.0;

contract IdentityContract {
    mapping (address => bytes32) public identities;

    function createIdentity(bytes32 _identity) public {
        require(_identity != 0, "Invalid identity");
        identities[msg.sender] = _identity;
    }

    function verifyIdentity(address _address, bytes32 _identity) public view returns (bool) {
        return identities[_address] == _identity;
    }
}
