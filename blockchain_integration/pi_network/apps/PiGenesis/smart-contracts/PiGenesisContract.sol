pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract PiGenesis is ERC20, Ownable {
    // ... (existing code)

    // New features
    uint256 public totalStakedTokens;
    mapping(address => uint256) public userStakedTokens;
    mapping(address => uint256) public userMiningReward;

    function stake(uint256 amount) external {
        require(amount > 0, "Amount must be greater than zero");
        require(amount <= balanceOf(msg.sender), "Insufficient balance");

        totalStakedTokens += amount;
        userStakedTokens[msg.sender] += amount;
        _burn(amount);
    }

    function unstake(uint256 amount) external {
        require(amount > 0, "Amount must be greater than zero");
        require(amount <= userStakedTokens[msg.sender], "Insufficient staked balance");

        totalStakedTokens -= amount;
        userStakedTokens[msg.sender] -= amount;
        _mint(msg.sender, amount);
    }

    function getStakingReward() external view returns (uint256) {
        // Implement staking reward calculation based on total staked tokens
    }

    function getUserStakingReward(address user) external view returns (uint256) {
        // Implement user staking reward calculation based on user's staked balance
    }

    function getUserMiningReward(address user) external view returns (uint256) {
        return userMiningReward[user];
    }

    function getUserLastMinedAt(address user) external view returns (uint256) {
        return lastMinedAt[user];
    }

    function getUserBalance(address user) external view override returns(uint256) {
        return balanceOf(user);
    }
}
