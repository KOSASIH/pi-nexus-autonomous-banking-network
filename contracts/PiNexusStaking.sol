pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiNexusStaking is SafeERC20 {
    // Staking properties
    uint256 public stakingReward;
    uint256 public stakingPeriod;

    // Staking constructor
    constructor() public {
        stakingReward = 10;
        stakingPeriod = 30 days;
    }

    // Staking functions
    function stake(uint256 amount) public {
        // Stake tokens to earn rewards
        _stake(msg.sender, amount);
    }

    function unstake() public {
        // Unstake tokens and claim rewards
        _unstake(msg.sender);
    }

    function claimReward() public {
        // Claim staking rewards
        _claimReward(msg.sender);
    }
}
