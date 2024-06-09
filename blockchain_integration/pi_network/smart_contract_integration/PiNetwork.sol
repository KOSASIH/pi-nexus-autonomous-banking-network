pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiNetwork {
    // Mapping of user addresses to their balances
    mapping (address => uint256) public balances;

    // Mapping of user addresses to their transaction history
    mapping (address => Transaction[]) public transactionHistory;

    // Event emitted when a user makes a transaction
    event TransactionEvent(address indexed from, address indexed to, uint256 value);

    // Event emitted when a user's balance is updated
    event BalanceUpdateEvent(address indexed user, uint256 newBalance);

    // Struct to represent a transaction
    struct Transaction {
        address from;
        address to;
        uint256 value;
        uint256 timestamp;
    }

    // Function to transfer tokens from one user to another
    function transfer(address to, uint256 value) public {
        require(balances[msg.sender] >= value, "Insufficient balance");
        balances[msg.sender] -= value;
        balances[to] += value;
        emit TransactionEvent(msg.sender, to, value);
    }

    // Function to get a user's balance
    function getBalance(address user) public view returns (uint256) {
        return balances[user];
    }

    // Function to get a user's transaction history
    function getTransactionHistory(address user) public view returns (Transaction[] memory) {
        return transactionHistory[user];
    }
}
