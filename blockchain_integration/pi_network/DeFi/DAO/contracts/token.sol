pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract Token {
    // Mapping of users to their balances
    mapping (address => uint256) public balances;

    // Event emitted when a user's balance is updated
    event BalanceUpdated(address user, uint256 balance);

    // Function to mint tokens
    function mint(address user, uint256 amount) public {
        // Check if the user is the owner or has the required permission
        require(msg.sender == owner || balances[msg.sender] >= amount, "Unauthorized");

        // Update the user's balance
        balances[user] += amount;

        // Emit the BalanceUpdated event
        emit BalanceUpdated(user, balances[user]);
    }

    // Function to burn tokens
    function burn(address user, uint256 amount) public {
        // Check if the user is the owner or has the required permission
        require(msg.sender == owner || balances[msg.sender] >= amount, "Unauthorized");

        // Update the user's balance
        balances[user] -= amount;

        // Emit the BalanceUpdated event
        emit BalanceUpdated(user, balances[user]);
    }
}
