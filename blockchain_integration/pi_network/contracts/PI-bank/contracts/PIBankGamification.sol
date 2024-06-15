pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankGamification {
    // Mapping of gamification rewards
    mapping(address => uint256) public gamificationRewards;

    // Event
    event NewGamificationReward(address indexed user, uint256 amount);

    // Function
    function claimGamificationReward(address user, uint256 amount) public {
        // Update gamification rewards
        gamificationRewards[user] = amount;
        emit NewGamificationReward(user, amount);
    }
}
