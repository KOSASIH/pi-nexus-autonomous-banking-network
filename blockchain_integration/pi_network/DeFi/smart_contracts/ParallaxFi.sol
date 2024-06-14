pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract ParallaxFi {
    // Events
    event ParallaxShift(uint256 indexed shift, uint256 timestamp);
    event DimensionalReward(address indexed user, uint256 amount);
    event CelestialDeposit(address indexed user, uint256 amount);
    event GalacticWithdrawal(address indexed user, uint256 amount);
    event ParallaxSwap(address indexed user, uint256 amount, address token);

    // Constants
    uint256 public constant PARALLAX_SHIFT_INTERVAL = 120 days; // 120-day parallax shift interval
    uint256 public constant DIMENSIONAL_REWARD_RATE = 0.60 ether; // 60% of total supply
    uint256 public constant CELESTIAL_DEPOSIT_FEE = 0.04 ether; // 4% deposit fee
    uint256 public constant GALACTIC_WITHDRAWAL_FEE = 0.05 ether; // 5% withdrawal fee
    uint256 public constant PARALLAX_SWAP_FEE = 0.015 ether; // 1.5% swap fee
    uint256 public constant MAX_SUPPLY = 10000000 ether; // 10 million tokens

    // State variables
    uint256 public totalSupply;
    uint256 public parallaxShift;
    uint256 public lastParallaxShiftTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public dimensionalRewards;
    mapping(address => mapping(address => uint256)) public parallaxSwaps;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        parallaxShift = 0;
        lastParallaxShiftTimestamp = block.timestamp;
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
        emit GalacticWithdrawal(msg.sender, amount);
    }

    function claimDimensionalReward() public {
        require(dimensionalRewards[msg.sender] > 0, "No dimensional reward available");
        uint256 reward = dimensionalRewards[msg.sender];
        dimensionalRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit DimensionalReward(msg.sender, reward);
    }

    function parallaxSwap(address token, uint256 amount) public {
        require(amount > 0, "Swap amount must be greater than 0");
        require(parallaxSwaps[msg.sender][token] >= amount, "Insufficient swap balance");
        parallaxSwaps[msg.sender][token] -= amount;
        balances[msg.sender] += amount;
        emit ParallaxSwap(msg.sender, amount, token);
    }

    function updateParallaxShift() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastParallaxShiftTimestamp >= PARALLAX_SHIFT_INTERVAL) {
            parallaxShift++;
            lastParallaxShiftTimestamp = currentTimestamp;
            emit ParallaxShift(parallaxShift, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 dimensionalRewardAmount = totalSupply * DIMENSIONAL_REWARD_RATE / 100;
        for (address user in balances) {
            dimensionalRewards[user] += dimensionalRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrative functions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setParallaxShiftInterval(uint256 newInterval) public onlyOwner {
        PARALLAX_SHIFT_INTERVAL = newInterval;
    }

   function setDimensionalRewardRate(uint256 newRate) public onlyOwner {
        DIMENSIONAL_REWARD_RATE = newRate;
    }

    function setCelestialDepositFee(uint256 newFee) public onlyOwner {
        CELESTIAL_DEPOSIT_FEE = newFee;
    }

    function setGalacticWithdrawalFee(uint256 newFee) public onlyOwner {
        GALACTIC_WITHDRAWAL_FEE = newFee;
    }

    function setParallaxSwapFee(uint256 newFee) public onlyOwner {
        PARALLAX_SWAP_FEE = newFee;
    }
}
