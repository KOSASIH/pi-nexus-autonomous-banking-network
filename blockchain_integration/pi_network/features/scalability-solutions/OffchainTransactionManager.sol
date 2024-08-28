pragma solidity ^0.8.0;

contract OffchainTransactionManager {
    // Mapping of off-chain transactions
    mapping (uint256 => OffchainTransaction) public offchainTransactions;

    // Struct to represent an off-chain transaction
    struct OffchainTransaction {
        uint256 transactionId;
        address sender;
        address recipient;
        uint256 amount;
        bytes data;
    }

    // Event emitted when an off-chain transaction is created
    event OffchainTransactionCreated(uint256 transactionId, address sender, address recipient, uint256 amount);

    // Function to create an off-chain transaction
    function createOffchainTransaction(address _sender, address _recipient, uint256 _amount, bytes memory _data) public {
        OffchainTransaction storage transaction = offchainTransactions[uint256(keccak256(abi.encodePacked(_sender, _recipient, _amount, _data)))];
        transaction.transactionId = uint256(keccak256(abi.encodePacked(_sender, _recipient, _amount, _data)));
        transaction.sender = _sender;
        transaction.recipient = _recipient;
        transaction.amount = _amount;
        transaction.data = _data;
        emit OffchainTransactionCreated(transaction.transactionId, _sender, _recipient, _amount);
    }

    // Function to get an off-chain transaction by ID
    function getOffchainTransaction(uint256 _transactionId) public view returns (OffchainTransaction memory) {
        return offchainTransactions[_transactionId];
    }
}
