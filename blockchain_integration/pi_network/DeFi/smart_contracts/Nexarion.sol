pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract Nexarion {
    // Events
    event AuroraBloom(uint256 indexed bloom, uint256 timestamp);
    event LuminousReward(address indexed user, uint256 amount);
    event CelestialDeposit(address indexed user, uint256 amount);
    event StellarWithdrawal(address indexed user, uint256 amount);
    event NexarionSwap(address indexed user, uint256 amount, address token);

    // Constants
    uint256 public constant AURORA_BLOOM_INTERVAL = 60 days; // 60-day aurora bloom interval
    uint256 public constant LUMINOUS_REWARD_RATE = 0.40 ether; // 40% of total supply
    uint256 public constant CELESTIAL_DEPOSIT_FEE = 0.02 ether; // 2% deposit fee
    uint256 public constant STELLAR_WITHDRAWAL_FEE = 0.03 ether; // 3% withdrawal fee
    uint256 public constant NEXARION_SWAP_FEE = 0.0075 ether; // 0.75% swap fee
    uint256 public constant MAX_SUPPLY = 10000000 ether; // 10 million tokens

    // State variables
    uint256 public totalSupply;
    uint256 public auroraBloom;
    uint256 public lastAuroraBloomTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public luminousRewards;
    mapping(address => mapping(address => uint256)) public nexarionSwaps;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        auroraBloom = 0;
        lastAuroraBloomTimestamp = block.timestamp;
        owner = msg.sender;
    }

    // Functions
    function deposit(uint256 amount) public {
        require(amount > 0, "Deposit amount must be greater than 0");
        balances[msg.sender] += amount;
        totalSupply += amount;
        emit CelestialDeposit(msg.sender, amount);
    }

    function withdraw(uint256 amount) public {
        require(amount > 0, "Withdrawal amount must be greater than 0");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        totalSupply -= amount;
        emit StellarWithdrawal(msg.sender, amount);
    }

    function claimLuminousReward() public {
        require(luminousRewards[msg.sender] > 0, "No luminous reward available");
        uint256 reward = luminousRewards[msg.sender];
        luminousRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit LuminousReward(msg.sender, reward);
    }

    function nexarionSwap(address token, uint256 amount) public {
        require(amount > 0, "Swap amount must be greater than 0");
        require(nexarionSwaps[msg.sender][token] >= amount, "Insufficient swap balance");
        nexarionSwaps[msg.sender][token] -= amount;
        balances[msg.sender] += amount;
        emit NexarionSwap(msg.sender, amount, token);
    }

    function updateAuroraBloom() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastAuroraBloomTimestamp >= AURORA_BLOOM_INTERVAL) {
            auroraBloom++;
            lastAuroraBloomTimestamp = currentTimestamp;
            emit AuroraBloom(auroraBloom, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 luminousRewardAmount = totalSupply * LUMINOUS_REWARD_RATE / 100;
        for (address user in balances) {
            luminousRewards[user] += luminousRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrativefunctions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setAuroraBloomInterval(uint256 newInterval) public onlyOwner {
        AURORA_BLOOM_INTERVAL = newInterval;
    }

    function setLuminousRewardRate(uint256 newRate) public onlyOwner {
        LUMINOUS_REWARD_RATE = newRate;
    }

    function setCelestialDepositFee(uint256 newFee) public onlyOwner {
        CELESTIAL_DEPOSIT_FEE = newFee;
    }

    function setStellarWithdrawalFee(uint256 newFee) public onlyOwner {
        STELLAR_WITHDRAWAL_FEE = newFee;
    }

    function setNexarionSwapFee(uint256 newFee) public onlyOwner {
        NEXARION_SWAP_FEE = newFee;
    }
}
