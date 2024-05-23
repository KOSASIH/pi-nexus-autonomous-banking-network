pragma solidity ^0.8.0;

contract MultiSigWallet {

    // The owner of the wallet
    address public owner;

    // The required number of signatures to authorize a transaction
    uint public requiredSignatures;

    // The list of owners of the wallet
    address[] public owners;

    // The list of pending transactions
    struct PendingTransaction {
        address to;
        uint value;
        bytes data;
        uint requiredSignatures;
        uint[] signatures;
        bool executed;
    }
    PendingTransaction[] public pendingTransactions;

    // The constructor function
    constructor(address[] memory _owners, uint _requiredSignatures) {
        owner = msg.sender;
        owners = _owners;
        requiredSignatures = _requiredSignatures;
    }

    // The function to submit a new transaction
    function submitTransaction(address _to, uint _value, bytes memory _data) external {
        require(msg.sender == owner, "Only the owner can submit transactions");
        require(requiredSignatures > 0, "Required signatures must be greater than 0");
        require(requiredSignatures <= owners.length, "Required signatures must be less than or equal to the number of owners");

        uint transactionId = pendingTransactions.length;
        pendingTransactions.push(PendingTransaction({
            to: _to,
            value: _value,
            data: _data,
            requiredSignatures: requiredSignatures,
            signatures: new uint[](requiredSignatures),
            executed: false
        }));

        emit TransactionSubmitted(transactionId);
    }

    // The function to authorize a transaction
    function authorizeTransaction(uint _transactionId) external {
        require(isOwner(msg.sender), "Only owners can authorize transactions");
        require(_transactionId < pendingTransactions.length, "Transaction not found");

        PendingTransaction storage transaction = pendingTransactions[_transactionId];
        require(!transaction.executed, "Transaction already executed");

        transaction.signatures.push(uint(keccak256(abi.encodePacked(msg.sender, block.timestamp))));

        uint requiredSignatures = transaction.requiredSignatures;
        uint signatures = transaction.signatures.length;

        if (signatures >= requiredSignatures) {
            executeTransaction(_transactionId);
        }
    }

    // The function to execute a transaction
    function executeTransaction(uint _transactionId) internal {
        PendingTransaction storage transaction = pendingTransactions[_transactionId];
        require(!transaction.executed, "Transaction already executed");

        (bool success, ) = transaction.to.call{value: transaction.value}(transaction.data);
        require(success, "Transaction execution failed");

        transaction.executed = true;

        emit TransactionExecuted(transactionId);
    }

    // The function to check if an address is an owner of the wallet
    function isOwner(address _address) public view returns (bool) {
        for (uint i = 0; i < owners.length; i++) {
            if (owners[i] == _address) {
                return true;
            }
        }
        return false;
    }

    // The event for when a transaction is submitted
    event TransactionSubmitted(uint indexed transactionId);

    // The event for when a transaction is executed
    event TransactionExecuted(uint indexed transactionId);
}
