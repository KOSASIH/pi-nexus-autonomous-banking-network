pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract DecentralizedFinance is Ownable {
    // Mapping of users to their respective balances
    mapping(address => uint256) public balances;

    // Mapping of assets to their respective prices
    mapping(address => uint256) public prices;

    // Event emitted when a user deposits funds
    event Deposit(address indexed user, uint256 amount);

    // Event emitted when a user withdraws funds
    event Withdrawal(address indexed user, uint256 amount);

    // Event emitted when a user trades an asset
    event Trade(address indexed user, address indexed asset, uint256 amount, uint256 price);

    // Function to deposit funds
    function deposit(uint256 amount) public {
        // Update the user's balance
        balances[msg.sender] += amount;

        // Emit the Deposit event
        emit Deposit(msg.sender, amount);
    }

    // Function to withdraw funds
    function withdraw(uint256 amount) public {
        // Check if the user has sufficient balance
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // Update the user's balance
        balances[msg.sender] -= amount;

        // Emit the Withdrawal event
        emit Withdrawal(msg.sender, amount);
    }

    // Function to trade an asset
    function trade(address asset, uint256 amount) public {
        // Check if the user has sufficient balance
        require(balances[msg.sender] >= amount * prices[asset], "Insufficient balance");

        // Update the user's balance
        balances[msg.sender] -= amount * prices[asset];

        // Emit the Trade event
        emit Trade(msg.sender, asset, amount, prices[asset]);
    }

    // Function to set the price of an asset
    function setPrice(address asset, uint256 price) public onlyOwner {
        // Update the price of the asset
        prices[asset] = price;
    }
}
