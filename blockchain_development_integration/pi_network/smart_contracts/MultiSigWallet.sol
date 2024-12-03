// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Address.sol";

contract MultiSigWallet is ReentrancyGuard {
    using Address for address;

    // State variables
    address[] public owners;
    mapping(address => bool) public isOwner;
    uint256 public requiredConfirmations;

    struct Transaction {
        address to;
        uint256 value;
        bytes data;
        bool executed;
        uint256 confirmations;
        mapping(address => bool) isConfirmed;
    }

    Transaction[] public transactions;

    // Events
    event Deposit(address indexed sender, uint256 amount);
    event TransactionCreated(uint256 indexed txIndex, address indexed to, uint256 value, bytes data);
    event TransactionConfirmed(uint256 indexed txIndex, address indexed owner);
    event TransactionExecuted(uint256 indexed txIndex);
    event OwnerAdded(address indexed newOwner);
    event OwnerRemoved(address indexed removedOwner);
    event RequiredConfirmationsChanged(uint256 requiredConfirmations);

    // Modifiers
    modifier onlyOwner() {
        require(isOwner[msg.sender], "Not an owner");
        _;
    }

    modifier txExists(uint256 txIndex) {
        require(txIndex < transactions.length, "Transaction does not exist");
        _;
    }

    modifier notExecuted(uint256 txIndex) {
        require(!transactions[txIndex].executed, "Transaction already executed");
        _;
    }

    modifier notConfirmed(uint256 txIndex) {
        require(!transactions[txIndex].isConfirmed[msg.sender], "Transaction already confirmed");
        _;
    }

    // Constructor
    constructor(address[] memory _owners, uint256 _requiredConfirmations) {
        require(_owners.length > 0, "Owners required");
        require(_requiredConfirmations > 0 && _requiredConfirmations <= _owners.length, "Invalid number of required confirmations");

        for (uint256 i = 0; i < _owners.length; i++) {
            address owner = _owners[i];
            require(owner != address(0), "Invalid owner");
            require(!isOwner[owner], "Owner not unique");

            isOwner[owner] = true;
            owners.push(owner);
        }
        requiredConfirmations = _requiredConfirmations;
    }

    // Fallback function to receive Ether
    receive() external payable {
        emit Deposit(msg.sender, msg.value);
    }

    // Create a new transaction
    function createTransaction(address to, uint256 value, bytes memory data) public onlyOwner {
        uint256 txIndex = transactions.length;
        transactions.push(Transaction({
            to: to,
            value: value,
            data: data,
            executed: false,
            confirmations: 0
        }));

        emit TransactionCreated(txIndex, to, value, data);
    }

    // Confirm a transaction
    function confirmTransaction(uint256 txIndex) public onlyOwner txExists(txIndex) notConfirmed(txIndex) notExecuted(txIndex) {
        Transaction storage transaction = transactions[txIndex];
        transaction.isConfirmed[msg.sender] = true;
        transaction.confirmations += 1;

        emit TransactionConfirmed(txIndex, msg.sender);
    }

    // Execute a confirmed transaction
    function executeTransaction(uint256 txIndex) public onlyOwner txExists(txIndex) notExecuted(txIndex) {
        Transaction storage transaction = transactions[txIndex];
        require(transaction.confirmations >= requiredConfirmations, "Not enough confirmations");

        transaction.executed = true;
        (bool success, ) = transaction.to.call{value: transaction.value}(transaction.data);
        require(success, "Transaction execution failed");

        emit TransactionExecuted(txIndex);
    }

    // Add a new owner
    function addOwner(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Invalid owner");
        require(!isOwner[newOwner], "Owner already exists");

        isOwner[newOwner] = true;
        owners.push(newOwner);

        emit OwnerAdded(newOwner);
    }

    // Remove an existing owner
    function removeOwner(address owner) public onlyOwner {
        require(isOwner[owner], "Not an owner");

        isOwner[owner] = false;
        for (uint256 i = 0; i < owners.length; i++) {
            if (owners[i] == owner) {
                owners[i] = owners[owners.length - 1];
                owners.pop();
                break;
            }
        }

        emit OwnerRemoved(owner);
    }

    // Change the number of required confirmations
    function changeRequiredConfirmations(uint256 _requiredConfirmations) public onlyOwner {
        require(_requiredConfirmations > 0 && _requiredConfirmations <= owners.length, "Invalid number of required confirmations");
        requiredConfirmations = _requiredConfirmations;

        emit RequiredConfirmationsChanged(_requiredConfirmations);
    }

    // Get the number of transactions
    function getTransactionCount() public view returns (uint256) {
        return transactions.length;
    }

    // Get details of a specific transaction
    function getTransaction(uint256 txIndex) public view returns (address to, uint256 value, bytes memory data, bool executed, uint256 confirmations) {
        Transaction storage transaction = transactions[txIndex];
        return (transaction.to, transaction.value, transaction.data, transaction.executed, transaction.confirmations);
    }

    // Get the list of owners
    function getOwners() public view returns (address[] memory) {
        return owners;
    }
}
