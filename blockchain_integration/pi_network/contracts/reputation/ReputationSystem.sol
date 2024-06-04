pragma solidity ^0.8.0;

contract ReputationSystem {
    mapping(address => uint256) public reputation;

    function updateReputation(address user, uint256 amount) public {
        reputation[user] += amount;
    }

    function getReputation(address user) public view returns (uint256) {
        return reputation[user];
    }
}
