pragma solidity ^0.8.0;

contract DataStorage {
    using ScalabilityOptimizer for address;

    // Mapping of transaction data
    mapping (uint256 => bytes) public transactionData;

    // Function to store transaction data
    function setTransactionData(uint256 _transactionId, bytes memory _data) public {
        transactionData[_transactionId] = _data;
    }

    // Function to retrieve transaction data
    function getTransactionData(uint256 _transactionId) public view returns (bytes memory) {
        return transactionData[_transactionId];
    }
}
