// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract MultiSigWallet is Ownable {
    using SafeMath for uint256;

    event Deposit(address indexed sender, uint256 amount);
    event SubmitTransaction(address indexed owner, uint256 indexed txIndex);
    event ConfirmTransaction(address indexed owner, uint256 indexed txIndex);
    event RevokeConfirmation(address indexed owner, uint256 indexed txIndex);
    event ExecuteTransaction(address indexed owner, uint256 indexed txIndex);

    struct Transaction {
        address to;
        uint256 value;
        bytes data;
        bool executed;
        uint256 confirmations;
    }

    address[] public owners;
    mapping(address => bool) public isOwner;
    mapping(uint256 => mapping(address => bool)) public isConfirmed;
    Transaction[] public transactions;
    uint256 public requiredConfirmations;

    modifier onlyOwner() {
        require(isOwner[msg.sender], "Not an owner");
        _;
    }

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

    // Fallback function to accept Ether
    receive() external payable {
        emit Deposit(msg.sender, msg.value);
    }

    // Function to submit a transaction
    function submitTransaction(address to, uint256 value, bytes memory data) public onlyOwner {
        uint256 txIndex = transactions.length;
        transactions.push(Transaction({
            to: to,
            value: value,
            data: data,
            executed: false,
            confirmations: 0
        }));

        emit SubmitTransaction(msg.sender, txIndex);
    }

    // Function to confirm a transaction
    function confirmTransaction(uint256 txIndex) public onlyOwner {
        require(txIndex < transactions.length, "Transaction does not exist");
        require(!isConfirmed[txIndex][msg.sender], "Transaction already confirmed");

        isConfirmed[txIndex][msg.sender] = true;
        transactions[txIndex].confirmations = transactions[txIndex].confirmations.add(1);

        emit ConfirmTransaction(msg.sender, txIndex);
        executeTransaction(txIndex);
    }

    // Function to revoke confirmation
    function revokeConfirmation(uint256 txIndex) public onlyOwner {
        require(txIndex < transactions.length, "Transaction does not exist");
        require(isConfirmed[txIndex][msg.sender], "Transaction not confirmed");

        isConfirmed[txIndex][msg.sender] = false;
        transactions[txIndex].confirmations = transactions[txIndex].confirmations.sub(1);

        emit RevokeConfirmation(msg.sender, txIndex);
    }

    // Function to execute a transaction
    function executeTransaction(uint256 txIndex) public {
        require(txIndex < transactions.length, "Transaction does not exist");
        Transaction storage transaction = transactions[txIndex];
        require(transaction.confirmations >= requiredConfirmations, "Not enough confirmations");
        require(!transaction.executed, "Transaction already executed");

        transaction.executed = true;

        (bool success, ) = transaction.to.call{value: transaction.value}(transaction.data);
        require(success, "Transaction failed");

        emit ExecuteTransaction(msg.sender, txIndex);
    }

    // Function to get the transaction count
    function getTransactionCount() public view returns (uint256) {
        return transactions.length;
    }

    // Function to get transaction details
    function getTransaction(uint256 txIndex) public view returns (
        address to,
        uint256 value,
        bytes memory data,
        bool executed,
        uint256 confirmations
    ) {
        require(txIndex < transactions.length, "Transaction does not exist");
        Transaction storage transaction = transactions[txIndex];
        return (
            transaction.to,
            transaction.value,
            transaction.data,
            transaction.executed,
            transaction.confirmations
        );
    }

    // Function to get the list of owners
    function getOwners() public view returns (address[] memory) {
        return owners;
    }
}
