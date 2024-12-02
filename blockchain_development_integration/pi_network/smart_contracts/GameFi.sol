// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract GameFi {
    IERC20 public token;
    mapping(address => uint256) public playerRewards;

    event RewardEarned(address indexed player, uint256 amount);

    constructor(IERC20 _token) {
        token = _token;
    }

    function playGame() external {
        // Simulate game logic
        uint256 reward = calculateReward(msg.sender);
        playerRewards[msg.sender] += reward;
        emit RewardEarned(msg.sender, reward);
    }

    function claimRewards() external {
        uint256 reward = playerRewards[msg.sender];
        require(reward > 0, "No rewards to claim");

        playerRewards[msg.sender] = 0;
        token.transfer(msg.sender, reward);
    }

    function calculateReward(address player) internal view returns (uint256) {
        // Implement your reward calculation logic here
        return 10; // Placeholder logic
    }
}
