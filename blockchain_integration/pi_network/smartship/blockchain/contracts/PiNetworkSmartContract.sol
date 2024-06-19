pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract PiNetworkSmartContract {
    using SafeERC20 for address;
    using SafeMath for uint256;

    // Mapping of user accounts to their balances
    mapping (address => uint256) public balances;

    // Mapping of transaction history
    mapping (address => Transaction[]) public transactionHistory;

    // Event emitted when a new account is created
    event NewAccount(address indexed user, uint256 balance);

    // Event emitted when a transaction is made
    event TransactionMade(address indexed from, address indexed to, uint256 amount);

    // Struct to represent a transaction
    struct Transaction {
        address from;
        address to;
        uint256 amount;
        uint256 timestamp;
    }

    // Function to create a new account
    function createAccount(address user, uint256 initialBalance) public {
        require(user!= address(0), "Invalid user address");
        require(initialBalance > 0, "Initial balance must be greater than 0");

        balances[user] = initialBalance;
        emit NewAccount(user, initialBalance);
    }

    // Function to transfer funds
    function transferFunds(address from, address to, uint256 amount) public {
        require(from!= address(0), "Invalid from address");
        require(to!= address(0), "Invalid to address");
        require(amount > 0, "Amount must be greater than 0");
        require(balances[from] >= amount, "Insufficient balance");

        balances[from] = balances[from].sub(amount);
        balances[to] = balances[to].add(amount);

        Transaction memory transaction = Transaction(from, to, amount, block.timestamp);
        transactionHistory[from].push(transaction);
        transactionHistory[to].push(transaction);

        emit TransactionMade(from, to, amount);
    }

    // Function to get account balance
    function getAccountBalance(address user) public view returns (uint256) {
        return balances[user];
    }

    // Function to get transaction history
    function getTransactionHistory(address user) public view returns (Transaction[] memory) {
        return transactionHistory[user];
    }
}
