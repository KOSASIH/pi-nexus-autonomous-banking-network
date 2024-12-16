// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract StakingContract is Ownable {
    using SafeMath for uint256;

    IERC20 public quantumCoin; // The QuantumCoin token contract
    uint256 public rewardRate; // Reward rate per second
    uint256 public totalStaked; // Total amount of staked tokens

    struct Stake {
        uint256 amount; // Amount of tokens staked
        uint256 timestamp; // Timestamp of the last stake
        uint256 rewards; // Accumulated rewards
    }

    mapping(address => Stake) public stakes; // Mapping of user stakes

    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardPaid(address indexed user, uint256 reward);

    constructor(IERC20 _quantumCoin, uint256 _rewardRate) {
        quantumCoin = _quantumCoin;
        rewardRate = _rewardRate;
    }

    // Function to stake tokens
    function stake(uint256 amount) external {
        require(amount > 0, "Cannot stake 0");
        require(quantumCoin.balanceOf(msg.sender) >= amount, "Insufficient balance");

        // Update the user's rewards before staking
        updateRewards(msg.sender);

        // Transfer tokens from the user to the contract
        quantumCoin.transferFrom(msg.sender, address(this), amount);

        // Update the user's stake
        stakes[msg.sender].amount = stakes[msg.sender].amount.add(amount);
        stakes[msg.sender].timestamp = block.timestamp;
        totalStaked = totalStaked.add(amount);

        emit Staked(msg.sender, amount);
    }

    // Function to unstake tokens and claim rewards
    function unstake(uint256 amount) external {
        require(amount > 0, "Cannot unstake 0");
        require(stakes[msg.sender].amount >= amount, "Insufficient staked balance");

        // Update the user's rewards before unstaking
        updateRewards(msg.sender);

        // Update the user's stake
        stakes[msg.sender].amount = stakes[msg.sender].amount.sub(amount);
        totalStaked = totalStaked.sub(amount);

        // Transfer tokens back to the user
        quantumCoin.transfer(msg.sender, amount);

        emit Unstaked(msg.sender, amount);
    }

    // Function to claim rewards
    function claimRewards() external {
        updateRewards(msg.sender);
        uint256 reward = stakes[msg.sender].rewards;
        require(reward > 0, "No rewards to claim");

        // Reset the user's rewards
        stakes[msg.sender].rewards = 0;

        // Transfer rewards to the user
        quantumCoin.transfer(msg.sender, reward);

        emit RewardPaid(msg.sender, reward);
    }

    // Function to update rewards for a user
    function updateRewards(address user) internal {
        Stake storage userStake = stakes[user];
        if (userStake.amount > 0) {
            uint256 timeStaked = block.timestamp.sub(userStake.timestamp);
            uint256 reward = userStake.amount.mul(rewardRate).mul(timeStaked).div(1e18);
            userStake.rewards = userStake.rewards.add(reward);
            userStake.timestamp = block.timestamp; // Reset the timestamp
        }
    }

    // Function to get the user's staked amount
    function stakedAmount(address user) external view returns (uint256) {
        return stakes[user].amount;
    }

    // Function to get the user's accumulated rewards
    function accumulatedRewards(address user) external view returns (uint256) {
        Stake storage userStake = stakes[user];
        uint256 timeStaked = block.timestamp.sub(userStake.timestamp);
        uint256 reward = userStake.amount.mul(rewardRate).mul(timeStaked).div(1e18);
        return userStake.rewards.add(reward);
    }

    // Function to set a new reward rate (only owner)
    function setRewardRate(uint256 newRate) external onlyOwner {
        rewardRate = newRate;
    }

    // Function to withdraw tokens in case of emergency (only owner)
    function emergencyWithdraw(uint256 amount) external onlyOwner {
        require(amount <= quantumCoin.balanceOf(address(this)), "Insufficient balance in contract");
        quantumCoin.transfer(msg.sender, amount);
    }
}
