pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract Galaxion {
    // Events
    event NewGalaxionCycle(uint256 indexed cycle, uint256 timestamp);
    event CosmicBoost(address indexed user, uint256 amount);
    event StellarReward(address indexed user, uint256 amount);
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event GalaxionSwap(address indexed user, uint256 amount, address token);

    // Constants
    uint256 public constant GALAXION_CYCLE_DURATION = 90 days; // 90-day galaxion cycle
    uint256 public constant COSMIC_BOOST_RATE = 0.20 ether; // 20% of total supply
    uint256 public constant STELLAR_REWARD_RATE = 0.15 ether; // 15% of total supply
    uint256 public constant MAX_SUPPLY = 10000000 ether; // 10 million tokens
    uint256 public constant GALAXION_SWAP_FEE = 0.005 ether; // 0.5% swap fee

    // State variables
    uint256 public totalSupply;
    uint256 public galaxionCycle;
    uint256 public lastGalaxionCycleTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public cosmicBoosts;
    mapping(address => uint256) public stellarRewards;
    mapping(address => mapping(address => uint256)) public galaxionSwaps;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        galaxionCycle = 0;
        lastGalaxionCycleTimestamp = block.timestamp;
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

    function claimCosmicBoost() public {
        require(cosmicBoosts[msg.sender] > 0, "No cosmic boost available");
        uint256 boost = cosmicBoosts[msg.sender];
        cosmicBoosts[msg.sender] = 0;
        balances[msg.sender] += boost;
        emit CosmicBoost(msg.sender, boost);
    }

    function claimStellarReward() public {
        require(stellarRewards[msg.sender] > 0, "No stellar reward available");
        uint256 reward = stellarRewards[msg.sender];
        stellarRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit StellarReward(msg.sender, reward);
    }

    function galaxionSwap(address token, uint256 amount) public {
        require(amount > 0, "Swap amount must be greater than 0");
        require(galaxionSwaps[msg.sender][token] >= amount, "Insufficient swap balance");
        galaxionSwaps[msg.sender][token] -= amount;
        balances[msg.sender] += amount;
        emit GalaxionSwap(msg.sender, amount, token);
    }

    function updateGalaxionCycle() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastGalaxionCycleTimestamp >= GALAXION_CYCLE_DURATION) {
            galaxionCycle++;
            lastGalaxionCycleTimestamp = currentTimestamp;
            emit NewGalaxionCycle(galaxionCycle, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 cosmicBoostAmount = totalSupply * COSMIC_BOOST_RATE / 100;
        uint256 stellarRewardAmount = totalSupply * STELLAR_REWARD_RATE / 100;
        for (address user in balances) {
            cosmicBoosts[user] += cosmicBoostAmount * balances[user] / totalSupply;
            stellarRewards[user] += stellarRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrative functions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setGalaxionCycleDuration(uint256 newDuration) public onlyOwner {
        GALAXION_CYCLE_DURATION = newDuration;
    }

    function setCosmicBoostRate(uint256 newRate) public onlyOwner {
        COSMIC_BOOST_RATE = newRate;
    }

    function setStellarRewardRate(uint256 newRate) public onlyOwner {
        STELLAR_REWARD_RATE = newRate;
    }

    function setGalaxionSwapFee(uint256 newFee) public onlyOwner {
        GALAXION_SWAP_FEE = newFee;
    }
}
