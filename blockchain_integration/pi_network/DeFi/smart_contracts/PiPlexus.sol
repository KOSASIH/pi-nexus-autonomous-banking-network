pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";
import "https://github.com/smartcontractkit/chainlink/blob/master/evm-contracts/src/v0.6/interfaces/AggregatorV3Interface.sol";
import "https://github.com/aave/protocol-v2/blob/master/contracts/protocol/lendingpool/LendingPool.sol";

contract PiPlexus {
    using SafeERC20 for IERC20;
    using SafeMath for uint256;

    // Mapping of user addresses to their balances
    mapping (address => uint256) public balances;

    // Mapping of asset addresses to their metadata
    mapping (address => Asset) public assets;

    // Mapping of user addresses to their staked assets
    mapping (address => mapping (address => uint256)) public stakes;

    // Mapping of user addresses to their rewards
    mapping (address => mapping (address => uint256)) public rewards;

    // Mapping of asset addresses to their prices
    mapping (address => uint256) public prices;

    // Mapping of user addresses to their collateral assets
    mapping (address => mapping (address => uint256)) public collaterals;

    // Mapping of user addresses to their debt assets
    mapping (address => mapping (address => uint256)) public debts;

    // Event emitted when a user deposits assets
    event Deposit(address indexed user, address indexed asset, uint256 amount);

    // Event emitted when a user withdraws assets
    event Withdrawal(address indexed user, address indexed asset, uint256 amount);

    // Event emitted when a user trades assets
    event Trade(address indexed user, address indexed assetIn, address indexed assetOut, uint256 amountIn, uint256 amountOut);

    // Event emitted when a user stakes assets
    event Stake(address indexed user, address indexed asset, uint256 amount);

    // Event emitted when a user unstakes assets
    event Unstake(address indexed user, address indexed asset, uint256 amount);

    // Event emitted when a user borrows assets
    event Borrow(address indexed user, address indexed asset, uint256 amount);

    // Event emitted when a user repays assets
    event Repay(address indexed user, address indexed asset, uint256 amount);

    // Event emitted when a user claims rewards
    event Claim(address indexed user, address indexed asset, uint256 amount);

    // Event emitted when a user liquidates a position
    event Liquidate(address indexed user, address indexed asset, uint256 amount);

    // Struct to represent an asset
    struct Asset {
        string name;
        string symbol;
        uint256 totalSupply;
        uint256 decimals;
        address[] holders;
    }

    // Chainlink Aggregator V3 Interface
    AggregatorV3Interface internal priceFeed;

    // Aave LendingPool Interface
    LendingPool internal lendingPool;

    // Admin address
    address public admin;

    // Maximum loan-to-value ratio
    uint256 public maxLTV;

    // Liquidation bonus
    uint256 public liquidationBonus;

    // Constructor
    constructor(address aggregator, address lendingPoolAddress, uint256 _maxLTV, uint256 _liquidationBonus) public {
        admin = msg.sender;
        priceFeed = AggregatorV3Interface(aggregator);
        lendingPool = LendingPool(lendingPoolAddress);
        maxLTV = _maxLTV;
        liquidationBonus = _liquidationBonus;
    }

    // Function to deposit assets
    function deposit(address asset, uint256 amount) public {
        require(amount > 0, "Invalid amount");
        require(asset.isContract(), "Asset must be a contract");
        IERC20(asset).safeTransferFrom(msg.sender, address(this), amount);
        balances[msg.sender] += amount;
        assets[asset].holders.push(msg.sender);
        emit Deposit(msg.sender, asset, amount);
    }

    // Function to withdraw assets
    function withdraw(address asset, uint256 amount) public {
        require(amount > 0, "Invalid amount");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        require(IERC20(asset).transfer(msg.sender, amount), "Transfer failed");
        balances[msg.sender] -= amount;
        emit Withdrawal(msg.sender, asset, amount);
    }

    //Function to trade assets
    function trade(address assetIn, address assetOut, uint256 amountIn) public {
        require(amountIn > 0, "Invalid amount");
        require(balances[msg.sender] >= amountIn, "Insufficient balance");
        uint256 amountOut = getAmountOut(assetIn, assetOut, amountIn);
        balances[msg.sender] -= amountIn;
        balances[msg.sender] += amountOut;
        emit Trade(msg.sender, assetIn, assetOut, amountIn, amountOut);
    }

    // Function to stake assets
    function stake(address asset, uint256 amount) public {
        require(amount > 0, "Invalid amount");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        stakes[msg.sender][asset] += amount;
        emit Stake(msg.sender, asset, amount);
    }

    // Function to unstake assets
    function unstake(address asset, uint256 amount) public {
        require(amount > 0, "Invalid amount");
        require(stakes[msg.sender][asset] >= amount, "Insufficient stake");
        stakes[msg.sender][asset] -= amount;
        emit Unstake(msg.sender, asset, amount);
    }

    // Function to borrow assets
    function borrow(address asset, uint256 amount) public {
        require(amount > 0, "Invalid amount");
        require(amount <= getBorrowable(msg.sender, asset), "Insufficient borrowable amount");
        require(collaterals[msg.sender][asset] >= amount, "Insufficient collateral");
        debts[msg.sender][asset] += amount;
        emit Borrow(msg.sender, asset, amount);
    }

    // Function to repay assets
    function repay(address asset, uint256 amount) public {
        require(amount > 0, "Invalid amount");
        require(debts[msg.sender][asset] >= amount, "Insufficient debt");
        debts[msg.sender][asset] -= amount;
        emit Repay(msg.sender, asset, amount);
    }

    // Function to claim rewards
    function claim(address asset) public {
        uint256 reward = getReward(msg.sender, asset);
        require(reward > 0, "No rewards available");
        rewards[msg.sender][asset] -= reward;
        balances[msg.sender] += reward;
        emit Claim(msg.sender, asset, reward);
    }

    // Function to get the amount out of a trade
    function getAmountOut(address assetIn, address assetOut, uint256 amountIn) internal view returns (uint256) {
        uint256 priceIn = getPrice(assetIn);
        uint256 priceOut = getPrice(assetOut);
        return amountIn * priceOut / priceIn;
    }

    // Function to get the borrowable amount for a user
    function getBorrowable(address user, address asset) internal view returns (uint256) {
        uint256 collateralValue = getCollateralValue(user, asset);
        uint256 debtValue = getDebtValue(user, asset);
        return collateralValue * maxLTV / 100 - debtValue;
    }

    // Function to get the collateral value for a user
    function getCollateralValue(address user, address asset) internal view returns (uint256) {
        uint256 collateralAmount = collaterals[user][asset];
        uint256 price = getPrice(asset);
        return collateralAmount * price;
    }

    // Function to get the debt value for a user
    function getDebtValue(address user, address asset) internal view returns (uint256) {
        uint256 debtAmount = debts[user][asset];
        uint256 price = getPrice(asset);
        return debtAmount * price;
    }

    // Function to get the reward for staking
    function getReward(address user, address asset) internal view returns (uint256) {
        uint256 totalStaked = getTotalStaked(asset);
        uint256 userStaked = stakes[user][asset];
        uint256 reward = (userStaked * 1000) / totalStaked;
        return reward;
    }

    // Function to get the total staked for an asset
    function getTotalStaked(address asset) internal view returns (uint256) {
        uint256 total = 0;
        for (address user : assets[asset].holders) {
            total += stakes[user][asset];
        }
        return total;
    }

    // Function to get the price of an asset
    function getPrice(address asset) internal view returns (uint256) {
        (, int256 price,,, ) = priceFeed.latestRoundData();
        return uint256(price);
    }

    // Function to update the price of an asset
    function updatePrice(address asset, uint256 newPrice) public {
        require(msg.sender == admin, "Only admin can update prices");
        prices[asset] = newPrice;
    }

    // Function to liquidate a user's position
    function liquidate(address user, address asset) public {
        require(debts[user][asset] > 0, "No debt to liquidate");
        uint256 debtValue = getDebtValue(user, asset);
        uint256 collateralValue = getCollateralValue(user, asset);
        if (debtValue > collateralValue * maxLTV / 100) {
            // Liquidate the position
            uint256 amountToLiquidate = debtValue - collateralValue * maxLTV / 100;
            debts[user][asset] -= amountToLiquidate;
            collaterals[user][asset] -= amountToLiquidate;
            emit Liquidate(user, asset, amountToLiquidate);
        }
    }

    // Function to lend assets
    function lend(address asset, uint256 amount) public {
        require(amount > 0, "Invalid amount");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        lendingPool.lend(asset, amount);
        emit Lend(msg.sender, asset, amount);
    }

    // Function to withdraw lent assets
    function withdrawLent(address asset, uint256 amount) public {
        require(amount > 0, "Invalid amount");
        require(lendingPool.getLentAmount(asset, msg.sender) >= amount, "Insufficient lent amount");
        lendingPool.withdrawLent(asset, amount);
        emit WithdrawLent(msg.sender, asset, amount);
    }
}
