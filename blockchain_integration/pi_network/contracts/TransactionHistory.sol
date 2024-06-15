pragma solidity ^0.8.0;

import "./PIBank.sol";

contract TransactionHistory {
    // Mapping of transaction history
    mapping (address => Transaction[]) public transactions;

    // Event emitted when a new transaction is added
    event NewTransaction(address indexed from, address indexed to, uint256 amount);

    // Function to add a new transaction
    function addTransaction(address from, address to, uint256 amount) public {
        transactions[from].push(Transaction(to, amount));
        emit NewTransaction(from, to, amount);
    }

    // Function to get transaction history
    function getTransactionHistory(address user) public view returns (Transaction[] memory) {
        return transactions[user];
    }
}

struct Transaction {
    address to;
    uint256 amount;
}
