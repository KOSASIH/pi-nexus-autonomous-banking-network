pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract Stellaluna {
    // Events
    event NewLunarCycle(uint256 indexed cycle, uint256 timestamp);
    event StellarReward(address indexed user, uint256 amount);
    event LunarReward(address indexed user, uint256 amount);
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);

    // Constants
    uint256 public constant LUNAR_CYCLE_DURATION = 30 days; // 30-day lunar cycle
    uint256 public constant STELLAR_REWARD_RATE = 0.05 ether; // 5% of total supply
    uint256 public constant LUNAR_REWARD_RATE = 0.03 ether; // 3% of total supply
    uint256 public constant MAX_SUPPLY = 1000000 ether; // 1 million tokens

    // State variables
    uint256 public totalSupply;
    uint256 public lunarCycle;
    uint256 public lastLunarCycleTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public stellarRewards;
    mapping(address => uint256) public lunarRewards;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        lunarCycle = 0;
        lastLunarCycleTimestamp = block.timestamp;
        owner = msg.sender;
    }

    // Functions
    function deposit(uint256 amount) public {
        require(amount > 0, "Deposit amount must be greater than 0");
        balances[msg.sender] += amount;
        totalSupply += amount;
        emit Deposit(msg.sender, amount);
    }

    function withdraw(uint256 amount) public {
        require(amount > 0, "Withdrawal amount must be greater than 0");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        totalSupply -= amount;
        emit Withdrawal(msg.sender, amount);
    }

    function claimStellarReward() public {
        require(stellarRewards[msg.sender] > 0, "No stellar reward available");
        uint256 reward = stellarRewards[msg.sender];
        stellarRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit StellarReward(msg.sender, reward);
    }

    function claimLunarReward() public {
        require(lunarRewards[msg.sender] > 0, "No lunar reward available");
        uint256 reward = lunarRewards[msg.sender];
        lunarRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit LunarReward(msg.sender, reward);
    }

    function updateLunarCycle() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastLunarCycleTimestamp >= LUNAR_CYCLE_DURATION) {
            lunarCycle++;
            lastLunarCycleTimestamp = currentTimestamp;
            emit NewLunarCycle(lunarCycle, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 stellarRewardAmount = totalSupply * STELLAR_REWARD_RATE / 100;
        uint256 lunarRewardAmount = totalSupply * LUNAR_REWARD_RATE / 100;
        for (address user in balances) {
            stellarRewards[user] += stellarRewardAmount * balances[user] / totalSupply;
            lunarRewards[user] += lunarRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrative functions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setLunarCycleDuration(uint256 newDuration) public onlyOwner {
        LUNAR_CYCLE_DURATION = newDuration;
    }

    function setStellarRewardRate(uint256 newRate) public onlyOwner {
        STELLAR_REWARD_RATE = newRate;
    }

    function setLunarRewardRate(uint256 newRate) public onlyOwner {
        LUNAR_REWARD_RATE = newRate;
    }
}
