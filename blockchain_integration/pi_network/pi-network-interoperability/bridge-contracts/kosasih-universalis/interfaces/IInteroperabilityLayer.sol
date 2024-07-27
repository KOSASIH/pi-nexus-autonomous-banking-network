pragma solidity ^0.8.0;

interface IInteroperabilityLayer {
    function executeTransaction(bytes _transactionData) external;
}
