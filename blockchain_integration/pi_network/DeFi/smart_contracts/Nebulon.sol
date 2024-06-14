pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract Nebulon {
    // Events
    event NewNebulaCycle(uint256 indexed cycle, uint256 timestamp);
    event StellarIgnition(address indexed user, uint256 amount);
    event GalacticReward(address indexed user, uint256 amount);
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event NebulaSwap(address indexed user, uint256 amount, address token);

    // Constants
    uint256 public constant NEBULA_CYCLE_DURATION = 90 days; // 90-day nebula cycle
    uint256 public constant STELLAR_IGNITION_RATE = 0.10 ether; // 10% of total supply
    uint256 public constant GALACTIC_REWARD_RATE = 0.05 ether; // 5% of total supply
    uint256 public constant MAX_SUPPLY = 10000000 ether; // 10 million tokens
    uint256 public constant NEBULA_SWAP_FEE = 0.005 ether; // 0.5% swap fee

    // State variables
    uint256 public totalSupply;
    uint256 public nebulaCycle;
    uint256 public lastNebulaCycleTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public stellarIgnitions;
    mapping(address => uint256) public galacticRewards;
    mapping(address => mapping(address => uint256)) public nebulaSwaps;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        nebulaCycle = 0;
        lastNebulaCycleTimestamp = block.timestamp;
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

    function claimStellarIgnition() public {
        require(stellarIgnitions[msg.sender] > 0, "No stellar ignition available");
        uint256 ignition = stellarIgnitions[msg.sender];
        stellarIgnitions[msg.sender] = 0;
        balances[msg.sender] += ignition;
        emit StellarIgnition(msg.sender, ignition);
    }

    function claimGalacticReward() public {
        require(galacticRewards[msg.sender] > 0, "No galactic reward available");
        uint256 reward = galacticRewards[msg.sender];
        galacticRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit GalacticReward(msg.sender, reward);
    }

    function nebulaSwap(address token, uint256 amount) public {
        require(amount > 0, "Swap amount must be greater than 0");
        require(nebulaSwaps[msg.sender][token] >= amount, "Insufficient swap balance");
        nebulaSwaps[msg.sender][token] -= amount;
        balances[msg.sender] += amount;
        emit NebulaSwap(msg.sender, amount, token);
    }

    function updateNebulaCycle() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastNebulaCycleTimestamp >= NEBULA_CYCLE_DURATION) {
            nebulaCycle++;
            lastNebulaCycleTimestamp = currentTimestamp;
            emit NewNebulaCycle(nebulaCycle, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 stellarIgnitionAmount = totalSupply * STELLAR_IGNITION_RATE / 100;
        uint256 galacticRewardAmount =totalSupply * GALACTIC_REWARD_RATE / 100;
        for (address user in balances) {
            stellarIgnitions[user] += stellarIgnitionAmount * balances[user] / totalSupply;
            galacticRewards[user] += galacticRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrative functions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setNebulaCycleDuration(uint256 newDuration) public onlyOwner {
        NEBULA_CYCLE_DURATION = newDuration;
    }

    function setStellarIgnitionRate(uint256 newRate) public onlyOwner {
        STELLAR_IGNITION_RATE = newRate;
    }

    function setGalacticRewardRate(uint256 newRate) public onlyOwner {
        GALACTIC_REWARD_RATE = newRate;
    }

    function setNebulaSwapFee(uint256 newFee) public onlyOwner {
        NEBULA_SWAP_FEE = newFee;
    }
}
