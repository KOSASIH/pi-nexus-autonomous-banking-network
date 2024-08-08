// Lending.sol

pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/ReentrancyGuard.sol";

contract Lending {
    using SafeERC20 for address;

    // Mapping of lending pools
    mapping (address => LendingPool) public lendingPools;

    // Event emitted when a new lending pool is created
    event LendingPoolCreated(address indexed creator, address indexed asset, uint256 interestRate);

    // Event emitted when a lending pool is updated
    event LendingPoolUpdated(address indexed creator, address indexed asset, uint256 interestRate);

    // Event emitted when a lending pool is closed
    event LendingPoolClosed(address indexed creator, address indexed asset);

    // Struct to store lending pool metadata
    struct LendingPool {
        address asset;
        uint256 interestRate;
        uint256 totalSupply;
        uint256 totalBorrowed;
        mapping (address => uint256) borrowerBalances;
    }

    // Modifier to prevent reentrancy attacks
    modifier nonReentrant() {
        require(!_isReentrant, "Reentrancy detected");
        _isReentrant = true;
        _;
        _isReentrant = false;
    }

    // Create a new lending pool
    function createLendingPool(address asset, uint256 interestRate) public nonReentrant {
        require(asset != address(0), "Invalid asset");
        require(interestRate > 0, "Invalid interest rate");
        LendingPool storage pool = lendingPools[msg.sender];
        pool.asset = asset;
        pool.interestRate = interestRate;
        pool.totalSupply = 0;
        pool.totalBorrowed = 0;
        emit LendingPoolCreated(msg.sender, asset, interestRate);
    }

    // Deposit assets into a lending pool
    function deposit(address asset, uint256 amount) public nonReentrant {
        require(asset != address(0), "Invalid asset");
        require(amount > 0, "Invalid amount");
        LendingPool storage pool = lendingPools[msg.sender];
        require(pool.asset == asset, "Asset mismatch");
        pool.totalSupply += amount;
        asset.safeTransferFrom(msg.sender, address(this), amount);
    }

    // Borrow assets from a lending pool
    function borrow(address asset, uint256 amount) public nonReentrant {
        require(asset != address(0), "Invalid asset");
        require(amount > 0, "Invalid amount");
        LendingPool storage pool = lendingPools[msg.sender];
        require(pool.asset == asset, "Asset mismatch");
        require(pool.totalSupply >= amount, "Insufficient liquidity");
        pool.totalBorrowed += amount;
        asset.safeTransfer(msg.sender, amount);
        borrowerBalances[msg.sender] += amount;
    }

    // Repay borrowed assets
    function repay(address asset, uint256 amount) public nonReentrant {
        require(asset != address(0), "Invalid asset");
        require(amount > 0, "Invalid amount");
        LendingPool storage pool = lendingPools[msg.sender];
        require(pool.asset == asset, "Asset mismatch");
        require(borrowerBalances[msg.sender] >= amount, "Insufficient balance");
        pool.totalBorrowed -= amount;
        borrowerBalances[msg.sender] -= amount;
        asset.safeTransferFrom(msg.sender, address(this), amount);
    }

    // Close a lending pool
    function closeLendingPool(address asset) public nonReentrant {
        require(asset != address(0), "Invalid asset");
        LendingPool storage pool = lendingPools[msg.sender];
        require(pool.asset == asset, "Asset mismatch");
        require(pool.totalBorrowed == 0, "Pool is not fully repaid");
        delete lendingPools[msg.sender];
        emit LendingPoolClosed(msg.sender, asset);
    }
}
