pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract NovaSpire {
    // Events
    event NovaBurst(uint256 indexed burst, uint256 timestamp);
    event StellarReward(address indexed user, uint256 amount);
    event GalacticDeposit(address indexed user, uint256 amount);
    event CosmicWithdrawal(address indexed user, uint256 amount);
    event NovaSwap(address indexed user, uint256 amount, address token);

    // Constants
    uint256 public constant NOVA_BURST_INTERVAL = 30 days; // 30-day nova burst interval
    uint256 public constant STELLAR_REWARD_RATE = 0.30 ether; // 30% of total supply
    uint256 public constant GALACTIC_DEPOSIT_FEE = 0.01 ether; // 1% deposit fee
    uint256 public constant COSMIC_WITHDRAWAL_FEE = 0.02 ether; // 2% withdrawal fee
    uint256 public constant NOVA_SWAP_FEE = 0.005 ether; // 0.5% swap fee
    uint256 public constant MAX_SUPPLY = 10000000 ether; // 10 million tokens

    // State variables
    uint256 public totalSupply;
    uint256 public novaBurst;
    uint256 public lastNovaBurstTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public stellarRewards;
    mapping(address => mapping(address => uint256)) public novaSwaps;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        novaBurst = 0;
        lastNovaBurstTimestamp = block.timestamp;
        owner = msg.sender;
    }

    // Functions
    function deposit(uint256 amount) public {
        require(amount > 0, "Deposit amount must be greater than 0");
        balances[msg.sender] += amount;
        totalSupply += amount;
        emit GalacticDeposit(msg.sender, amount);
    }

    function withdraw(uint256 amount) public {
        require(amount > 0, "Withdrawal amount must be greater than 0");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        totalSupply -= amount;
        emit CosmicWithdrawal(msg.sender, amount);
    }

    function claimStellarReward() public {
        require(stellarRewards[msg.sender] > 0, "No stellar reward available");
        uint256 reward = stellarRewards[msg.sender];
        stellarRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit StellarReward(msg.sender, reward);
    }

    function novaSwap(address token, uint256 amount) public {
        require(amount > 0, "Swap amount must be greater than 0");
        require(novaSwaps[msg.sender][token] >= amount, "Insufficient swap balance");
        novaSwaps[msg.sender][token] -= amount;
        balances[msg.sender] += amount;
        emit NovaSwap(msg.sender, amount, token);
    }

    function updateNovaBurst() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastNovaBurstTimestamp >= NOVA_BURST_INTERVAL) {
            novaBurst++;
            lastNovaBurstTimestamp = currentTimestamp;
            emit NovaBurst(novaBurst, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 stellarRewardAmount = totalSupply * STELLAR_REWARD_RATE / 100;
        for (address user in balances) {
            stellarRewards[user] += stellarRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrative functions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setNovaBurstInterval(uint256 newInterval) public onlyOwner {
        NOVA_BURST_INTERVAL = newInterval;
    }

    function setStellarRewardRate(uint256 newRate) public onlyOwner {
        STELLAR_REWARD_RATE = newRate;
    }

    function setGalacticDepositFee(uint256 newFee) public onlyOwner {
        GALACTIC_DEPOSIT_FEE = newFee;
    }

    function setCosmicWithdrawalFee(uint256 newFee) public onlyOwner {
        COSMIC_WITHDRAWAL_FEE = newFee;
    }

    function setNovaSwapFee(uint256 newFee) public onlyOwner {
        NOVA_SWAP_FEE = newFee;
    }
}
