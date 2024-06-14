pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract CosmiaFi {
    // Events
    event NewGalacticCycle(uint256 indexed cycle, uint256 timestamp);
    event CosmicReward(address indexed user, uint256 amount);
    event StellarReward(address indexed user, uint256 amount);
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event GalacticSwap(address indexed user, uint256 amount, address token);

    // Constants
    uint256 public constant GALACTIC_CYCLE_DURATION = 60 days; // 60-day galactic cycle
    uint256 public constant COSMIC_REWARD_RATE = 0.07 ether; // 7% of total supply
    uint256 public constant STELLAR_REWARD_RATE = 0.04 ether; // 4% of total supply
    uint256 public constant MAX_SUPPLY = 10000000 ether; // 10 million tokens
    uint256 public constant GALACTIC_SWAP_FEE = 0.005 ether; // 0.5% swap fee

    // State variables
    uint256 public totalSupply;
    uint256 public galacticCycle;
    uint256 public lastGalacticCycleTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public cosmicRewards;
    mapping(address => uint256) public stellarRewards;
    mapping(address => mapping(address => uint256)) public galacticSwaps;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        galacticCycle = 0;
        lastGalacticCycleTimestamp = block.timestamp;
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

    function claimCosmicReward() public {
        require(cosmicRewards[msg.sender] > 0, "No cosmic reward available");
        uint256 reward = cosmicRewards[msg.sender];
        cosmicRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit CosmicReward(msg.sender, reward);
    }

    function claimStellarReward() public {
        require(stellarRewards[msg.sender] > 0, "No stellar reward available");
        uint256 reward = stellarRewards[msg.sender];
        stellarRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit StellarReward(msg.sender, reward);
    }

    function galacticSwap(address token, uint256 amount) public {
        require(amount > 0, "Swap amount must be greater than 0");
        require(galacticSwaps[msg.sender][token] >= amount, "Insufficient swap balance");
        galacticSwaps[msg.sender][token] -= amount;
        balances[msg.sender] += amount;
        emit GalacticSwap(msg.sender, amount, token);
    }

    function updateGalacticCycle() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastGalacticCycleTimestamp >= GALACTIC_CYCLE_DURATION) {
            galacticCycle++;
            lastGalacticCycleTimestamp = currentTimestamp;
            emit NewGalacticCycle(galacticCycle, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 cosmicRewardAmount = totalSupply * COSMIC_REWARD_RATE / 100;
        uint256 stellarRewardAmount = totalSupply * STELLAR_REWARD_RATE / 100;
        for (address user in balances) {
            cosmicRewards[user] += cosmicRewardAmount * balances[user] / totalSupply;
            stellarRewards[user] += stellarRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrative functions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setGalacticCycleDuration(uint256 newDuration) public onlyOwner {
        GALACTIC_CYCLE_DURATION = newDuration;
    }

    function setCosmicRewardRate(uint256 newRate) public onlyOwner {
        COSMIC_REWARD_RATE = newRate;
    }

    function setStellarRewardRate(uint256 newRate) public onlyOwner {
        STELLAR_REWARD_RATE = newRate;
    }

    function setGalacticSwapFee(uint256 newFee) public onlyOwner {
        GALACTIC_SWAP_FEE = newFee;
    }
}
