pragma solidity ^0.8.0;

contract IdentityManagement {
    mapping (address => string) public identities;
    mapping (address => uint256) public reputationScores;

    function registerIdentity(string memory _identity) public {
        identities[msg.sender] = _identity;
        reputationScores[msg.sender] = 0;
    }

    function updateReputationScore(address _address, uint256 _score) public {
        reputationScores[_address] = _score;
    }

    function getIdentity(address _address) public view returns (string memory) {
        return identities[_address];
    }

    function getReputationScore(address _address) public view returns (uint256) {
        return reputationScores[_address];
    }
}
