// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract YieldFarming is Ownable {
    struct Staker {
        uint256 amountStaked;
        uint256 rewardDebt;
    }

    IERC20 public stakingToken;
    IERC20 public rewardToken;
    uint256 public rewardPerBlock;
    uint256 public totalStaked;
    uint256 public lastRewardBlock;

    mapping(address => Staker) public stakers;

    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardClaimed(address indexed user, uint256 amount);

    constructor(IERC20 _stakingToken, IERC20 _rewardToken, uint256 _rewardPerBlock) {
        stakingToken = _stakingToken;
        rewardToken = _rewardToken;
        rewardPerBlock = _rewardPerBlock;
        lastRewardBlock = block.number;
    }

    // Update rewards
    function updateRewards() internal {
        if (block.number > lastRewardBlock) {
            uint256 blocks = block.number - lastRewardBlock;
            uint256 reward = blocks * rewardPerBlock;
            totalStaked += reward;
            lastRewardBlock = block.number;
        }
    }

    // Stake tokens
    function stake(uint256 _amount) public {
        updateRewards();
        Staker storage staker = stakers[msg.sender];

        if (staker.amountStaked > 0) {
            uint256 pendingReward = (staker.amountStaked * rewardPerBlock) - staker.rewardDebt;
            if (pendingReward > 0) {
                rewardToken.transfer(msg.sender, pendingReward);
                emit RewardClaimed(msg.sender, pendingReward);
            }
        }

        stakingToken.transferFrom(msg.sender, address(this), _amount);
        staker.amountStaked += _amount;
        staker.rewardDebt = staker.amountStaked * rewardPerBlock;
totalStaked += _amount;
        emit Staked(msg.sender, _amount);
    }

    // Unstake tokens
    function unstake(uint256 _amount) public {
        Staker storage staker = stakers[msg.sender];
        require(staker.amountStaked >= _amount, "Insufficient staked amount.");

        updateRewards();
        uint256 pendingReward = (staker.amountStaked * rewardPerBlock) - staker.rewardDebt;
        if (pendingReward > 0) {
            rewardToken.transfer(msg.sender, pendingReward);
            emit RewardClaimed(msg.sender, pendingReward);
        }

        staker.amountStaked -= _amount;
        stakingToken.transfer(msg.sender, _amount);
        emit Unstaked(msg.sender, _amount);
    }

    // Claim rewards
    function claimRewards() public {
        updateRewards();
        Staker storage staker = stakers[msg.sender];
        uint256 pendingReward = (staker.amountStaked * rewardPerBlock) - staker.rewardDebt;

        require(pendingReward > 0, "No rewards to claim.");
        rewardToken.transfer(msg.sender, pendingReward);
        staker.rewardDebt = staker.amountStaked * rewardPerBlock;
        emit RewardClaimed(msg.sender, pendingReward);
    }

    // Get staker details
    function getStakerDetails(address _staker) public view returns (uint256, uint256) {
        Staker storage staker = stakers[_staker];
        return (staker.amountStaked, staker.rewardDebt);
    }
}
