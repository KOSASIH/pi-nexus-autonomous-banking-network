// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract Staking {
    IERC20 public token;
    mapping(address => uint256) public stakedAmount;
    mapping(address => uint256) public rewards;

    constructor(IERC20 _token) {
        token = _token;
    }

    function stake(uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        token.transferFrom(msg.sender, address(this), amount);
        stakedAmount[msg.sender] += amount;
    }

    function withdraw(uint256 amount) external {
        require(stakedAmount[msg.sender] >= amount, "Insufficient staked amount");
        stakedAmount[msg.sender] -= amount;
        token.transfer(msg.sender, amount);
    }

    function calculateReward(address user) public view returns (uint256) {
        // Simple reward calculation (1% of staked amount)
        return stakedAmount[user] / 100;
    }

    function claimReward() external {
        uint256 reward = calculateReward(msg.sender);
        require(reward > 0, "No rewards available");
        rewards[msg.sender] += reward;
        token.transfer(msg.sender, reward);
    }
}
