pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract AuroraChain {
    // Events
    event NewAuroraCycle(uint256 indexed cycle, uint256 timestamp);
    event DawnBreaker(address indexed user, uint256 amount);
    event RadiantReward(address indexed user, uint256 amount);
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event AuroraSwap(address indexed user, uint256 amount, address token);

    // Constants
    uint256 public constant AURORA_CYCLE_DURATION = 120 days; // 120-day aurora cycle
    uint256 public constant DAWN_BREAKER_RATE = 0.15 ether; // 15% of total supply
    uint256 public constant RADIANT_REWARD_RATE = 0.10 ether; // 10% of total supply
    uint256 public constant MAX_SUPPLY = 10000000 ether; // 10 million tokens
    uint256 public constant AURORA_SWAP_FEE = 0.005 ether; // 0.5% swap fee

    // State variables
    uint256 public totalSupply;
    uint256 public auroraCycle;
    uint256 public lastAuroraCycleTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public dawnBreakers;
    mapping(address => uint256) public radiantRewards;
    mapping(address => mapping(address => uint256)) public auroraSwaps;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        auroraCycle = 0;
        lastAuroraCycleTimestamp = block.timestamp;
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

    function claimDawnBreaker() public {
        require(dawnBreakers[msg.sender] > 0, "No dawn breaker available");
        uint256 breaker = dawnBreakers[msg.sender];
        dawnBreakers[msg.sender] = 0;
        balances[msg.sender] += breaker;
        emit DawnBreaker(msg.sender, breaker);
    }

    function claimRadiantReward() public {
        require(radiantRewards[msg.sender] > 0, "No radiant reward available");
        uint256 reward = radiantRewards[msg.sender];
        radiantRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit RadiantReward(msg.sender, reward);
    }

    function auroraSwap(address token, uint256 amount) public {
        require(amount > 0, "Swap amount must be greater than 0");
        require(auroraSwaps[msg.sender][token] >= amount, "Insufficient swap balance");
        auroraSwaps[msg.sender][token] -= amount;
        balances[msg.sender] += amount;
        emit AuroraSwap(msg.sender, amount, token);
    }

    function updateAuroraCycle() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastAuroraCycleTimestamp >= AURORA_CYCLE_DURATION) {
            auroraCycle++;
            lastAuroraCycleTimestamp = currentTimestamp;
            emit NewAuroraCycle(auroraCycle, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 dawnBreakerAmount = totalSupply * DAWN_BREAKER_RATE / 100;
        uint256 radiantRewardAmount = totalSupply * RADIANT_REWARD_RATE/ 100;
        for (address user in balances) {
            dawnBreakers[user] += dawnBreakerAmount * balances[user] / totalSupply;
            radiantRewards[user] += radiantRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrative functions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setAuroraCycleDuration(uint256 newDuration) public onlyOwner {
        AURORA_CYCLE_DURATION = newDuration;
    }

    function setDawnBreakerRate(uint256 newRate) public onlyOwner {
        DAWN_BREAKER_RATE = newRate;
    }

    function setRadiantRewardRate(uint256 newRate) public onlyOwner {
        RADIANT_REWARD_RATE = newRate;
    }

    function setAuroraSwapFee(uint256 newFee) public onlyOwner {
        AURORA_SWAP_FEE = newFee;
    }
}
