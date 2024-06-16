pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PIPaymentGateway {
    using SafeMath for uint256;
    using SafeERC20 for ERC20;

    // Mapping of user addresses to their transaction histories
    mapping (address => mapping (address => Transaction[])) public transactionHistory;

    // Event emitted when a new transaction is created
    event TransactionCreated(address payer, address payee, uint256 amount, uint256 timestamp);

    // Function to make a payment to another user or merchant
    function makePayment(address payee, uint256 amount) public {
        require(amount > 0, "Invalid payment amount");
        ERC20(0x1234567890123456789012345678901234567890).safeTransferFrom(msg.sender, payee, amount);
        Transaction memory newTransaction = Transaction(msg.sender, payee, amount, block.timestamp);
        transactionHistory[msg.sender][payee].push(newTransaction);
        emit TransactionCreated(msg.sender, payee, amount, block.timestamp);
    }

    // Struct to represent a transaction
    struct Transaction {
        address payer;
        address payee;
        uint256 amount;
        uint256 timestamp;
    }
}
