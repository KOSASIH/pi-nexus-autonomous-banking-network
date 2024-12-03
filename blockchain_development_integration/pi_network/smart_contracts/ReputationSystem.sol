// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ReputationSystem {
    mapping(address => uint256) public reputations;

    event ReputationUpdated(address indexed user, uint256 newReputation);

    function increaseReputation(address user, uint256 amount) external {
        reputations[user] += amount;
        emit ReputationUpdated(user, reputations[user]);
    }

    function decreaseReputation(address user, uint256 amount) external {
        require(reputations[user] >= amount, "Insufficient reputation");
        reputations[user] -= amount;
        emit ReputationUpdated(user, reputations[user]);
    }

    function getReputation(address user) external view returns (uint256) {
        return reputations[user];
    }
}
