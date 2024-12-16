// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Snapshot.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract QuantumCoin is ERC20, Ownable, Pausable, ERC20Burnable, ERC20Pausable, ERC20Snapshot {
    using SafeMath for uint256;

    // Staking variables
    mapping(address => uint256) private _stakingBalance;
    mapping(address => uint256) private _stakingTimestamp;
    uint256 public rewardRate = 100; // Reward rate per second
    uint256 public totalStaked;

    // Events
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardPaid(address indexed user, uint256 reward);

    constructor(uint256 initialSupply) ERC20("QuantumCoin", "QTC") {
        _mint(msg.sender, initialSupply * (10 ** decimals()));
    }

    // Function to stake tokens
    function stake(uint256 amount) external whenNotPaused {
        require(amount > 0, "Cannot stake 0");
        _transfer(msg.sender, address(this), amount);
        _stakingBalance[msg.sender] = _stakingBalance[msg.sender].add(amount);
        _stakingTimestamp[msg.sender] = block.timestamp;
        totalStaked = totalStaked.add(amount);
        emit Staked(msg.sender, amount);
    }

    // Function to unstake tokens and claim rewards
    function unstake(uint256 amount) external whenNotPaused {
        require(amount > 0, "Cannot unstake 0");
        require(_stakingBalance[msg.sender] >= amount, "Insufficient staked balance");
        
        uint256 reward = calculateReward(msg.sender);
        _stakingBalance[msg.sender] = _stakingBalance[msg.sender].sub(amount);
        totalStaked = totalStaked.sub(amount);
        _transfer(address(this), msg.sender, amount);
        _mint(msg.sender, reward); // Mint reward tokens
        emit Unstaked(msg.sender, amount);
        emit RewardPaid(msg.sender, reward);
    }

    // Function to calculate rewards
    function calculateReward(address user) public view returns (uint256) {
        uint256 stakedTime = block.timestamp.sub(_stakingTimestamp[user]);
        return _stakingBalance[user].mul(rewardRate).mul(stakedTime).div(1e18);
    }

    // Function to pause the contract
    function pause() external onlyOwner {
        _pause();
    }

    // Function to unpause the contract
    function unpause() external onlyOwner {
        _unpause();
    }

    // Function to snapshot balances
    function snapshot() external onlyOwner {
        _snapshot();
    }

    // Function to get staking balance
    function stakingBalance(address user) external view returns (uint256) {
        return _stakingBalance[user];
    }

    // Function to get total staked amount
    function totalStakedAmount() external view returns (uint256) {
        return totalStaked;
    }

    // Override functions to include pausable functionality
    function _beforeTokenTransfer(address from, address to, uint256 amount) internal override(ERC20, ERC20Pausable) {
        super._beforeTokenTransfer(from, to, amount);
    }
}
