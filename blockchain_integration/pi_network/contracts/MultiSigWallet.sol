// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MultiSigWallet {
    event Deposit(address indexed sender, uint256 amount);
    event SubmitTransaction(address indexed owner, uint256 indexed txIndex);
    event ApproveTransaction(address indexed owner, uint256 indexed txIndex);
    event ExecuteTransaction(address indexed owner, uint256 indexed txIndex);

    address[] public owners;
    mapping(address => bool) public isOwner;
    uint256 public required;

    struct Transaction {
        address to;
        uint256 value;
        bool executed;
        uint256 approvalCount;
        mapping(address => bool) approved;
    }

    Transaction[] public transactions;

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

    constructor(address[] memory _owners, uint256 _required) {
        require(_owners.length > 0, "Owners required");
        require(_required > 0 && _required <= _owners.length, "Invalid required number of owners");

        for (uint256 i = 0; i < _owners.length; i++) {
            address owner = _owners[i];
            require(!isOwner[owner], "Owner not unique");
            isOwner[owner] = true;
            owners.push(owner);
        }
        required = _required;
    }

    receive() external payable {
        emit Deposit(msg.sender, msg.value);
    }

    function submitTransaction(address to, uint256 value) external onlyOwner {
        uint256 txIndex = transactions.length;
        transactions.push();
        Transaction storage t = transactions[txIndex];
        t.to = to;
        t.value = value;
        emit SubmitTransaction(msg.sender, txIndex);
    }

    function approveTransaction(uint256 txIndex) external onlyOwner txExists(txIndex) notExecuted(txIndex) {
        Transaction storage t = transactions[txIndex];
        require(!t.approved[msg.sender], "Transaction already approved");

        t.approved[msg.sender] = true;
        t.approvalCount++;
        emit ApproveTransaction(msg.sender, txIndex);
    }

    function executeTransaction(uint256 txIndex) external onlyOwner txExists(txIndex) notExecuted(txIndex) {
        Transaction storage t = transactions[txIndex];
        require(t.approvalCount >= required, "Not enough approvals");

        t.executed = true;
        (bool success, ) = t.to.call{value: t.value}("");
        require(success, "Transaction failed");
        emit ExecuteTransaction(msg.sender, txIndex);
    }
}
