pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract Compound {
    // Mapping of user balances
    mapping (address => uint256) public balances;

    // Mapping of asset prices
    mapping (address => uint256) public prices;

    // Event emitted when a user deposits assets
    event Deposit(address indexed user, uint256 amount);

    // Event emitted when a user withdraws assets
    event Withdrawal(address indexed user, uint256 amount);

    // Event emitted when a user borrows assets
    event Borrow(address indexed user, uint256 amount);

    // Event emitted when a user repays borrowed assets
    event Repay(address indexed user, uint256 amount);

    // Function to deposit assets
    function deposit(uint256 amount) public {
        require(amount > 0, "Invalid deposit amount");
        balances[msg.sender] += amount;
        emit Deposit(msg.sender, amount);
    }

    // Function to withdraw assets
    function withdraw(uint256 amount) public {
        require(amount > 0, "Invalid withdrawal amount");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        emit Withdrawal(msg.sender, amount);
    }

    // Function to borrow assets
    function borrow(uint256 amount) public {
        require(amount > 0, "Invalid borrow amount");
        require(prices[msg.sender] > 0, "Asset price not set");
        uint256 borrowAmount = amount * prices[msg.sender];
        require(balances[msg.sender] >= borrowAmount, "Insufficient balance");
        balances[msg.sender] -= borrowAmount;
        emit Borrow(msg.sender, amount);
    }

    // Function to repay borrowed assets
    function repay(uint256 amount) public {
        require(amount > 0, "Invalid repayment amount");
        require(prices[msg.sender] > 0, "Asset price not set");
        uint256 repayAmount = amount * prices[msg.sender];
        require(balances[msg.sender] >= repayAmount, "Insufficient balance");
        balances[msg.sender] += repayAmount;
        emit Repay(msg.sender, amount);
    }

    // Function to set asset prices
    function setPrice(address asset, uint256 price) public {
        prices[asset] = price;
    }
}
