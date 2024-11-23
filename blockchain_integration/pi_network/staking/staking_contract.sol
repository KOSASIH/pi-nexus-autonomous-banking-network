// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Staking {
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardPaid(address indexed user, uint256 reward);

    mapping(address => uint256) public stakes;
    mapping(address => uint256) public rewards;
    uint256 public totalStaked;
    uint256 public rewardRate; // Reward rate per second
    uint256 public lastUpdateTime;

    constructor(uint256 _rewardRate) {
        rewardRate = _rewardRate;
    }

    modifier updateReward(address user) {
        rewards[user] += earned(user);
        lastUpdateTime = block.timestamp;
        _;
    }

    function stake(uint256 amount) external updateReward(msg.sender) {
        require(amount > 0, "Cannot stake 0");
        stakes[msg.sender] += amount;
        totalStaked += amount;
        emit Staked(msg.sender, amount);
    }

    function unstake(uint256 amount) external updateReward(msg.sender) {
        require(amount > 0, "Cannot unstake 0");
        require(stakes[msg.sender] >= amount, "Insufficient staked amount");
        stakes[msg.sender] -= amount;
        totalStaked -= amount;
        emit Unstaked(msg.sender, amount);
    }

    function earned(address user) public view returns (uint256) {
        return (stakes[user] * rewardRate * (block.timestamp - lastUpdateTime)) / 1e18 + rewards[user];
    }

    function claimReward() external updateReward(msg.sender) {
        uint256 reward = rewards[msg.sender];
        require(reward > 0, "No reward available");
        rewards[msg.sender] = 0;
        emit RewardPaid(msg.sender, reward);
        // Transfer the reward to the user (implement reward transfer logic)
    }

    function getStake(address user) external view returns (uint256) {
        return stakes[user];
    }

    function getReward(address user) external view returns (uint256) {
        return earned(user);
    }
}
