// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MultiSigWallet {
    event Deposit(address indexed sender, uint amount);
    event SubmitTransaction(address indexed owner, uint indexed txIndex);
    event ApproveTransaction(address indexed owner, uint indexed txIndex);
    event ExecuteTransaction(address indexed owner, uint indexed txIndex);

    address[] public owners;
    mapping(address => bool) public isOwner;
    uint public required;
    
    struct Transaction {
        address to;
        uint value;
        bool executed;
        uint approvalCount;
        mapping(address => bool) approved;
    }

    Transaction[] public transactions;

    modifier onlyOwner() {
        require(isOwner[msg.sender], "Not an owner");
        _;
    }

    modifier txExists(uint txIndex) {
        require(txIndex < transactions.length, "Transaction does not exist");
        _;
    }

    modifier notExecuted(uint txIndex) {
        require(!transactions[txIndex].executed, "Transaction already executed");
        _;
    }

    constructor(address[] memory _owners, uint _required) {
        require(_owners.length > 0, "Owners required");
        require(_required > 0 && _required <= _owners.length, "Invalid required number of owners");

        for (uint i = 0; i < _owners.length; i++) {
            address owner = _owners[i];
            require(owner != address(0), "Invalid owner");
            require(!isOwner[owner], "Owner is not unique");
            isOwner[owner] = true;
            owners.push(owner);
        }
        required = _required;
    }

    receive() external payable {
        emit Deposit(msg.sender, msg.value);
    }

    function submitTransaction(address to, uint value) public onlyOwner {
        uint txIndex = transactions.length;
        transactions.push();
        Transaction storage t = transactions[txIndex];
        t.to = to;
        t.value = value;
        emit SubmitTransaction(msg.sender, txIndex);
    }

    function approveTransaction(uint txIndex) public onlyOwner txExists(txIndex) notExecuted(txIndex) {
        Transaction storage t = transactions[txIndex];
        require(!t.approved[msg.sender], "Transaction already approved");

        t.approved[msg.sender] = true;
        t.approvalCount += 1;
        emit ApproveTransaction(msg.sender, txIndex);
    }

    function executeTransaction(uint txIndex) public onlyOwner txExists(txIndex) notExecuted(txIndex) {
        Transaction storage t = transactions[txIndex];
        require(t.approvalCount >= required, "Not enough approvals");

        t.executed = true;
        (bool success, ) = t.to.call{value: t.value}("");
        require(success, "Transaction failed");
        emit ExecuteTransaction(msg.sender, txIndex);
    }

    function getTransactionCount() public view returns (uint) {
        return transactions.length;
    }

    function getTransaction(uint txIndex) public view returns (address, uint, bool, uint) {
        Transaction storage t = transactions[txIndex];
        return (t.to, t.value, t.executed, t.approvalCount);
    }
}
