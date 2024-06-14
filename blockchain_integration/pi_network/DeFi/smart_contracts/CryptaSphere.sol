pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract CryptaSphere {
    // Events
    event NewSphereCycle(uint256 indexed cycle, uint256 timestamp);
    event SecureVault(address indexed user, uint256 amount);
    event InclusiveReward(address indexed user, uint256 amount);
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event SphereSwap(address indexed user, uint256 amount, address token);

    // Constants
    uint256 public constant SPHERE_CYCLE_DURATION = 120 days; // 120-day sphere cycle
    uint256 public constant SECURE_VAULT_RATE = 0.25 ether; // 25% of total supply
    uint256 public constant INCLUSIVE_REWARD_RATE = 0.20 ether; // 20% of total supply
    uint256 public constant MAX_SUPPLY = 10000000 ether; // 10 million tokens
    uint256 public constant SPHERE_SWAP_FEE = 0.005 ether; // 0.5% swap fee

    // State variables
    uint256 public totalSupply;
    uint256 public sphereCycle;
    uint256 public lastSphereCycleTimestamp;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public secureVaults;
    mapping(address => uint256) public inclusiveRewards;
    mapping(address => mapping(address => uint256)) public sphereSwaps;

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Constructor
    constructor() public {
        totalSupply = 0;
        sphereCycle = 0;
        lastSphereCycleTimestamp = block.timestamp;
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

    function claimSecureVault() public {
        require(secureVaults[msg.sender] > 0, "No secure vault available");
        uint256 vault = secureVaults[msg.sender];
        secureVaults[msg.sender] = 0;
        balances[msg.sender] += vault;
        emit SecureVault(msg.sender, vault);
    }

    function claimInclusiveReward() public {
        require(inclusiveRewards[msg.sender] > 0, "No inclusive reward available");
        uint256 reward = inclusiveRewards[msg.sender];
        inclusiveRewards[msg.sender] = 0;
        balances[msg.sender] += reward;
        emit InclusiveReward(msg.sender, reward);
    }

    function sphereSwap(address token, uint256 amount) public {
        require(amount > 0, "Swap amount must be greater than 0");
        require(sphereSwaps[msg.sender][token] >= amount, "Insufficient swap balance");
        sphereSwaps[msg.sender][token] -= amount;
        balances[msg.sender] += amount;
        emit SphereSwap(msg.sender, amount, token);
    }

    function updateSphereCycle() internal {
        uint256 currentTimestamp = block.timestamp;
        if (currentTimestamp - lastSphereCycleTimestamp >= SPHERE_CYCLE_DURATION) {
            sphereCycle++;
            lastSphereCycleTimestamp = currentTimestamp;
            emit NewSphereCycle(sphereCycle, currentTimestamp);
            distributeRewards();
        }
    }

    function distributeRewards() internal {
        uint256 secureVaultAmount = totalSupply * SECURE_VAULT_RATE / 100;
        uint256 inclusiveRewardAmount = totalSupply * INCLUSIVE_REWARD_RATE / 100;
        for (address user in balances) {
            secureVaults[user] += secureVaultAmount * balances[user] / totalSupply;
            inclusiveRewards[user] += inclusiveRewardAmount * balances[user] / totalSupply;
        }
    }

    // Administrative functions
    function setOwner(address newOwner) public onlyOwner {
        owner = newOwner;
    }

    function setSphereCycleDuration(uint256 newDuration) public onlyOwner {
        SPHERE_CYCLE_DURATION = newDuration;
    }

    function setSecureVaultRate(uint256 newRate) public onlyOwner {
        SECURE_VAULT_RATE = newRate;
    }

    function setInclusiveRewardRate(uint256 newRate) public onlyOwner {
        INCLUSIVE_REWARD_RATE = newRate;
    }

    function setSphereSwapFee(uint256 newFee) public onlyOwner {
        SPHERE_SWAP_FEE = newFee;
    }
}
